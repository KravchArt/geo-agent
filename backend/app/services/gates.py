"""Parallel preflight gates for scope and censorship.

The orchestrator depends only on :class:`GateEvaluator`. The current implementations
are deterministic regex stubs; future small LLMs can replace them by implementing the
same async ``evaluate`` method and returning the same ``GateDecision`` contract.
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from backend.app.llm.base import LLMClient
from backend.app.prompts import CENSOR_SYSTEM_PROMPT, SCOPER_SYSTEM_PROMPT
from backend.app.services.scope_classifier import ScopeClassifierScorer
from backend.app.services.scope_ood import OODVerdict, ScopeOODDetector
from common.models import (
    ConversationTurn,
    GateDecision,
    GateName,
    GateResults,
    GateVerdict,
    LLMMessage,
    LLMRequest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GateRule:
    """One named regex rule used by a rule-based gate."""

    rule_id: str
    pattern: re.Pattern[str]


def _rules(items: dict[str, str]) -> tuple[GateRule, ...]:
    return tuple(
        GateRule(rule_id=rule_id, pattern=re.compile(pattern, re.IGNORECASE))
        for rule_id, pattern in items.items()
    )


SCOPE_RULES = _rules(
    {
        "travel_general": (
            r"\b(travel|trip|vacation|holiday|tourism|tourist|journey|"
            r"getaway|backpacking|adventure)\b"
        ),
        "itinerary_planning": (
            r"\b(itinerary|day[ -]?trip|travel plan|"
            r"plan (?:a|my|our|the)? ?(?:trip|vacation|holiday|journey)|"
            r"plan (?:one|two|three|four|\d+) day[s]?\b|"
            r"things to do|what to do)\b"
        ),
        "destination_planning": (
            r"\b(plan|create|build|make|organize|suggest|recommend)\b"
            r".{0,50}"
            r"\b(in|to|for|around)\b"
            r".{0,50}"
            r"\b[a-zA-Z]{3,}\b"
        ),
        "destination_navigation": (
            r"\b(route|directions?|navigate|walking route|driving route|"
            r"public transport|transit|how to get)\b"
        ),
        "nearby_places": (
            r"\b(?:nearest|nearby|close(?:st)?|around me|near me)\b"
            r".{0,40}"
            r"\b(?:cafe|coffee|restaurant|bar|museum|hotel|hostel|"
            r"attraction|park|beach|shop|pharmacy|station|airport)\b"
        ),
        "place_search": (
            r"\b(?:find|recommend|suggest|show|search for|best|top|"
            r"where can i find)\b"
            r".{0,50}"
            r"\b(?:hotel|hostel|resort|cafe|coffee shop|restaurant|bar|"
            r"museum|gallery|attraction|landmark|park|beach|market)\b"
        ),
        "accommodation": (
            r"\b(hotel|hostel|resort|accommodation|lodging|"
            r"where to stay|check[ -]?in|check[ -]?out)\b"
        ),
        "transport_booking": (
            r"\b(flight|airfare|airport|train|rail|bus ticket|ferry|"
            r"car rental|taxi|transfer)\b"
        ),
        "food_while_travelling": (
            r"\b(where to eat|local food|local cuisine|restaurant in|"
            r"cafe in|food tour|best cafes)\b"
        ),
        "sights_activities": (
            r"\b(sightseeing|attraction|landmark|museum|gallery|"
            r"guided tour|excursion|activity|things to see)\b"
        ),
        "travel_logistics": (
            r"\b(visa|passport|border crossing|customs|travel insurance|"
            r"tourist tax|currency exchange)\b"
        ),
        "place_information": (
            r"\b(opening hours?|opening times?|ticket price|"
            r"admission|reservation|is .* open)\b"
        ),
        "destination_weather": (
            r"\b(weather|forecast|temperature)\b"
            r".{0,25}"
            r"\b(in|for|at)\b"
        ),
        "geo_location": (
            r"\b(location|coordinates?|latitude|longitude|"
            r"map|distance between|near|nearby)\b"
        ),
        "city_visit": (
            r"\b(visit|explore|discover)\b"
            r".{0,50}"
            r"\b[a-zA-Z]{3,}\b"
        ),
    }
)


CENSORSHIP_RULES = _rules(
    {
        "weapon_construction": (
            r"\b(?:how to|instructions? to|guide to|steps? to|build|make|assemble|manufacture)\b"
            r".{0,45}\b(?:bomb|explosive|grenade|firearm|gun|silencer|weapon)\b"
        ),
        "explosive_materials": (
            r"\b(?:explosive recipe|homemade explosive|improvised explosive|ied|detonator)\b"
        ),
        "violent_harm": (
            r"\b(?:how to|best way to|help me|plan to)\b.{0,35}"
            r"\b(?:kill|murder|assassinate|poison|kidnap|torture|seriously hurt)\b"
        ),
        "terrorism": (
            r"\b(?:plan|support|join|fund|recruit for)\b.{0,35}"
            r"\b(?:terrorist|terrorism|extremist group)\b"
        ),
        "malware_creation": (
            r"\b(?:write|create|build|develop|deploy)\b.{0,40}"
            r"\b(?:malware|ransomware|keylogger|botnet|computer virus|credential stealer)\b"
        ),
        "unauthorized_hacking": (
            r"\b(?:hack|breach|exploit|break into|bypass authentication|steal credentials)\b"
            r".{0,50}\b(?:account|server|network|website|database|wifi|system)\b"
        ),
        "drug_manufacturing": (
            r"\b(?:how to|recipe for|synthesize|manufacture|cook|make)\b.{0,40}"
            r"\b(?:meth|methamphetamine|cocaine|heroin|fentanyl|illegal drugs?)\b"
        ),
        "sexual_minors": (
            r"\b(?:child|minor|underage|teen(?:ager)?)\b.{0,30}"
            r"\b(?:sexual|porn|nude|explicit)\b|"
            r"\b(?:sexual|porn|nude|explicit)\b.{0,30}\b(?:child|minor|underage)\b"
        ),
        "self_harm_instructions": (
            r"\b(?:how to|best way to|instructions? for|help me)\b.{0,35}"
            r"\b(?:commit suicide|kill myself|self[- ]harm|overdose)\b"
        ),
        "human_trafficking": (
            r"\b(?:buy|sell|traffic|smuggle)\b.{0,30}"
            r"\b(?:person|people|women|children|migrants)\b"
        ),
        "hate_violence": (
            r"\b(?:attack|kill|eliminate|exterminate|hurt)\b.{0,30}"
            r"\b(?:race|ethnic|religious|nationality|gay|trans|immigrants?)\b"
        ),
        "fraud_identity_theft": (
            r"\b(?:steal identity|identity theft|credit card fraud|forge passport|"
            r"fake passport|phishing kit)\b"
        ),
    }
)


class GateEvaluator(ABC):
    """Replaceable async gate interface."""

    name: GateName

    @abstractmethod
    async def evaluate(
        self, text: str, *, history: Sequence[ConversationTurn] = ()
    ) -> GateDecision:
        """Return a normalized decision without raising for an ordinary rejection.

        ``history`` is the recent dialogue, oldest first. A follow-up like "и что
        рядом?" or "второй" is in scope only because of what came before, so a
        gate that judges the message alone rejects legitimate turns. Gates that
        cannot use context (regex) simply ignore it.
        """
        raise NotImplementedError


class RuleBasedGate(GateEvaluator):
    """Base implementation for deterministic regex gates."""

    def __init__(
        self,
        *,
        name: GateName,
        rules: tuple[GateRule, ...],
        match_means_pass: bool,
        pass_response: str,
        reject_response: str,
        pass_reason: str,
        reject_reason: str,
    ) -> None:
        self.name = name
        self._rules = rules
        self._match_means_pass = match_means_pass
        self._pass_response = pass_response
        self._reject_response = reject_response
        self._pass_reason = pass_reason
        self._reject_reason = reject_reason

    async def evaluate(
        self, text: str, *, history: Sequence[ConversationTurn] = ()
    ) -> GateDecision:
        # Regex rules read the current message only; context cannot help them.
        started = perf_counter()
        matched = [rule.rule_id for rule in self._rules if rule.pattern.search(text)]
        has_match = bool(matched)
        passed = has_match if self._match_means_pass else not has_match
        return GateDecision(
            name=self.name,
            verdict=GateVerdict.ALLOW if passed else GateVerdict.REJECT,
            passed=passed,
            response=self._pass_response if passed else self._reject_response,
            reason=self._pass_reason if passed else self._reject_reason,
            provider="rule_based",
            model=None,
            confidence=1.0,
            matched_rules=matched,
            latency_ms=int((perf_counter() - started) * 1000),
        )


class RuleBasedScopeGate(RuleBasedGate):
    def __init__(self) -> None:
        super().__init__(
            name=GateName.SCOPE,
            rules=SCOPE_RULES,
            match_means_pass=True,
            pass_response="The request is travel-related and may proceed to the main agent.",
            reject_response="I can only help with travel and location-related questions.",
            pass_reason="At least one travel or geospatial intent rule matched.",
            reject_reason="No travel or geospatial intent rule matched.",
        )


class NoScoperScopeGate(GateEvaluator):
    """Disable scope filtering while preserving an explicit audit decision."""

    name = GateName.SCOPE

    async def evaluate(
        self, text: str, *, history: Sequence[ConversationTurn] = ()
    ) -> GateDecision:
        return GateDecision(
            name=self.name,
            verdict=GateVerdict.ALLOW,
            passed=True,
            response="Scope filtering is disabled; the request may proceed.",
            reason="Scope filtering is disabled by SCOPE_PROVIDER=no_scoper.",
            provider="no_scoper",
            model=None,
            confidence=1.0,
            matched_rules=[],
            latency_ms=0,
        )


class SessionUnlockedScopeGate(GateEvaluator):
    """Stands in for the scope gate on a session that already passed it once.

    Not a classifier: it records WHY the check was skipped so a session that drifts
    off topic after an in-scope opener is visible in the logs rather than silently
    unfiltered. See ``SCOPE_SESSION_UNLOCK``.
    """

    name = GateName.SCOPE

    async def evaluate(
        self, text: str, *, history: Sequence[ConversationTurn] = ()
    ) -> GateDecision:
        return GateDecision(
            name=self.name,
            verdict=GateVerdict.ALLOW,
            passed=True,
            response="The request is travel-related and may proceed to the main agent.",
            reason=(
                "Scope was not re-checked: this session already produced an in-scope "
                "verdict and SCOPE_SESSION_UNLOCK is enabled."
            ),
            provider="session_unlocked",
            model=None,
            confidence=1.0,
            matched_rules=[],
            latency_ms=0,
        )


class RuleBasedCensorshipGate(RuleBasedGate):
    def __init__(self) -> None:
        super().__init__(
            name=GateName.CENSORSHIP,
            rules=CENSORSHIP_RULES,
            match_means_pass=False,
            pass_response="No prohibited request pattern was detected.",
            reject_response="I cannot help with that request.",
            pass_reason="No prohibited-content rule matched.",
            reject_reason="One or more prohibited-content rules matched.",
        )


#: The verdict must be the LAST word of the reply: a bare "no", or a narrated
#: "...so yes". Anything else is the model answering the question instead of
#: classifying it — a weak model replies "Yes, I can help with that, the museums
#: in Kazan are..." and reading that as a verdict would pass everything.
_VERDICT_RE = re.compile(r"\b(yes|no)\b[\s.!,;:'\")»]*$", re.IGNORECASE)

#: The classifier only ever emits one word; keep room for a short preamble but
#: not for an essay.
_SCOPE_MAX_TOKENS = 16


#: How many earlier turns the classifier sees. Enough to resolve a follow-up,
#: short enough that the verdict still hinges on the CURRENT message.
_SCOPE_HISTORY_TURNS = 4


def _scope_messages(text: str, history: Sequence[ConversationTurn]) -> list[LLMMessage]:
    """Classifier prompt: recent dialogue as context, then the message to judge.

    The turn under judgement is labelled explicitly. Without that the model tends
    to classify the conversation as a whole, so one in-scope opener would keep
    letting later off-topic turns through.
    """
    messages = [LLMMessage(role="system", content=SCOPER_SYSTEM_PROMPT)]
    recent = list(history)[-_SCOPE_HISTORY_TURNS:]
    if recent:
        transcript = "\n".join(
            f"{turn.get('role', '?')}: {turn.get('content', '')}" for turn in recent
        )
        messages.append(
            LLMMessage(
                role="user",
                content=(
                    "Earlier turns of this dialogue, for context only — do not "
                    f"classify them:\n{transcript}"
                ),
            )
        )
    messages.append(LLMMessage(role="user", content=f"Classify this request:\n{text}"))
    return messages


def parse_scope_verdict(content: str) -> bool | None:
    """Return True/False for an in/out-of-scope reply, or None if unparseable."""
    match = _VERDICT_RE.search((content or "").strip())
    if match is None:
        return None
    return match.group(1).lower() == "yes"


#: Same last-word rule as the scope verdict. "unsafe" is matched before "safe"
#: because the alternation is ordered and "safe" is a substring of "unsafe" —
#: reading "unsafe" as a pass would let harmful text through.
_CENSOR_VERDICT_RE = re.compile(r"\b(unsafe|safe)\b[\s.!,;:'\")»]*$", re.IGNORECASE)

#: One word, same as the scope classifier.
_CENSOR_MAX_TOKENS = 16


def _censor_messages(text: str) -> list[LLMMessage]:
    """Classifier prompt for one piece of text, with no dialogue context.

    Deliberately history-free: the censorship gate also judges the model's own
    answer, and a rejected turn must not be excusable by what came before it.
    """
    return [
        LLMMessage(role="system", content=CENSOR_SYSTEM_PROMPT),
        LLMMessage(role="user", content=f"Classify this text:\n{text}"),
    ]


def parse_censor_verdict(content: str) -> bool | None:
    """Return True when the text is safe, False when unsafe, None if unparseable."""
    match = _CENSOR_VERDICT_RE.search((content or "").strip())
    if match is None:
        return None
    return match.group(1).lower() == "safe"


#: Granite Guardian answers with the risk, not the safety: ``Yes`` means the risk
#: IS present. Reading it like the safe/unsafe vocabulary would invert every
#: verdict, so it gets its own parser.
_GUARDIAN_VERDICT_RE = re.compile(r"\b(yes|no)\b[\s.!,;:'\")»]*$", re.IGNORECASE)


def parse_guardian_verdict(content: str) -> bool | None:
    """Return True when Guardian says the risk is absent, False when present."""
    match = _GUARDIAN_VERDICT_RE.search((content or "").strip())
    if match is None:
        return None
    return match.group(1).lower() == "no"


class LLMScopeGate(GateEvaluator):
    """Scope classification by an LLM, with the regex gate as a safety net.

    Uses ``SCOPER_SYSTEM_PROMPT``, which asks for a bare ``yes``/``no``. The regex
    gate can only match phrasings someone anticipated; a model generalises — it
    understands that "И что рядом?" continues a geo dialogue while "напиши
    сортировку на Python" does not, without either being in a rule list.

    Any failure (model down, timeout, unparseable answer) falls back to the
    deterministic gate instead of guessing: a request must always get a verdict,
    and silently allowing everything when the classifier dies is how out-of-scope
    traffic reaches the main model unnoticed.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str | None = None,
        fallback: GateEvaluator | None = None,
    ) -> None:
        self.name = GateName.SCOPE
        self._llm = llm
        self._model = model
        self._fallback = fallback or RuleBasedScopeGate()

    async def evaluate(
        self, text: str, *, history: Sequence[ConversationTurn] = ()
    ) -> GateDecision:
        started = perf_counter()
        try:
            response = await self._llm.generate(
                LLMRequest(
                    messages=_scope_messages(text, history),
                    model=self._model,
                    temperature=0.0,
                    max_tokens=_SCOPE_MAX_TOKENS,
                )
            )
        except Exception as exc:
            primary_latency_ms = int((perf_counter() - started) * 1000)
            logger.warning("scope_gate_llm_failed falling_back_to_rules", exc_info=True)
            primary = GateDecision(
                name=self.name,
                verdict=GateVerdict.REJECT,
                passed=False,
                response="The model-backed scope check failed.",
                reason="Primary scope classifier execution failed; rules fallback was used.",
                provider="model",
                model=self._model,
                confidence=1.0,
                latency_ms=primary_latency_ms,
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            fallback = await self._fallback.evaluate(text, history=history)
            return fallback.model_copy(update={"attempts": [primary, fallback]})

        passed = parse_scope_verdict(response.content)
        if passed is None:
            primary_latency_ms = int((perf_counter() - started) * 1000)
            logger.warning(
                "scope_gate_unparseable falling_back_to_rules content=%r",
                response.content[:200],
            )
            primary = GateDecision(
                name=self.name,
                verdict=GateVerdict.REJECT,
                passed=False,
                response="The model-backed scope result could not be parsed.",
                reason="Primary scope classifier returned an unparseable verdict;"
                "rules fallback was used.",
                provider="model",
                model=response.model,
                confidence=1.0,
                latency_ms=primary_latency_ms,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                success=False,
                error_type="UnparseableGateResponse",
                error_message=(response.content or "")[:1000],
            )
            fallback = await self._fallback.evaluate(text, history=history)
            return fallback.model_copy(update={"attempts": [primary, fallback]})

        return GateDecision(
            name=self.name,
            verdict=GateVerdict.ALLOW if passed else GateVerdict.REJECT,
            passed=passed,
            response=(
                "The request is travel-related and may proceed to the main agent."
                if passed
                else "I can only help with travel and location-related questions."
            ),
            reason=f"Scope classifier answered {'yes' if passed else 'no'}.",
            provider="model",
            model=response.model,
            # A bare yes/no carries no probability; the verdict itself is discrete.
            confidence=1.0,
            matched_rules=[],
            latency_ms=int((perf_counter() - started) * 1000),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )


class ClassifierScopeGate(GateEvaluator):
    """Fine-tuned encoder in front of the LLM scoper.

    The classifier answers in one forward pass on a small encoder, far cheaper
    than a chat completion. What it cannot do is judge a follow-up: it sees one
    string with no dialogue, so "а что рядом?" scores like an out-of-scope
    fragment even though the previous turn makes it obviously in scope.

    ``confirm_reject`` (off by default) trades cost for that blind spot: with it on,
    **the encoder may admit a request on its own, but may not turn one away on its
    own** — every rejection goes to ``fallback``, the LLM scoper that reads the
    recent turns, and only its verdict reaches the user. Letting a stray request
    through costs one orchestrator turn that finds no tools to use; turning a real
    user away costs the user. The price is one small chat completion per rejected
    turn, which is why it is opt-in rather than the default.

    A failed or unreachable classifier also falls through to ``fallback`` rather
    than guessing, so a dead sidecar degrades to the previous behaviour instead of
    letting everything past.
    """

    def __init__(
        self,
        client: ScopeClassifierScorer,
        fallback: GateEvaluator,
        *,
        grey_low: float,
        grey_high: float,
        ood: ScopeOODDetector | None = None,
        ood_enforce: bool = True,
        confirm_reject: bool = False,
    ) -> None:
        self.name = GateName.SCOPE
        self._client = client
        self._fallback = fallback
        self._confirm_reject = confirm_reject
        self._grey_low = grey_low
        self._grey_high = grey_high
        self._ood = ood
        self._ood_enforce = ood_enforce

    async def _ood_verdict(self, text: str) -> OODVerdict | None:
        """Distance to the training corpus, or ``None`` if unavailable.

        A dead embedding sidecar must not take the gate down with it: without the
        distance the gate degrades to exactly its previous behaviour, which is a
        working scope check, not an open door.
        """
        if self._ood is None:
            return None
        try:
            return await self._ood.evaluate(text)
        except Exception:
            logger.warning("scope_ood_failed continuing_without_distance", exc_info=True)
            return None

    def _decision(
        self,
        *,
        passed: bool,
        score: float,
        latency_ms: int,
        ood: OODVerdict | None = None,
    ) -> GateDecision:
        # In shadow mode the distance is recorded but never changes the verdict —
        # that is how you learn the real flag rate on live traffic before paying
        # for it. The rate measured offline (2.6% at the 1% quantile) comes from a
        # QA corpus whose topics sit close to the training data; open traffic is
        # more varied and will flag more.
        ood_note = (
            ""
            if ood is None
            else (
                f" Distance to corpus {ood.similarity:.3f} (quantile {ood.quantile:.1%})"
                f"{'; flagged, shadow mode' if ood.far and not self._ood_enforce else ''}."
            )
        )
        return GateDecision(
            name=self.name,
            verdict=GateVerdict.ALLOW if passed else GateVerdict.REJECT,
            passed=passed,
            response=(
                "The request is travel-related and may proceed to the main agent."
                if passed
                else "I can only help with travel and location-related questions."
            ),
            reason=(
                f"Scope classifier scored {score:.3f} "
                f"(grey zone {self._grey_low:.2f}–{self._grey_high:.2f}).{ood_note}"
            ),
            provider="model",
            model=self._client.model,
            # Unlike the yes/no gates this really is a probability, so report the
            # confidence in the verdict that was actually returned.
            confidence=score if passed else 1.0 - score,
            matched_rules=[],
            latency_ms=latency_ms,
        )

    async def _defer(
        self,
        text: str,
        history: Sequence[ConversationTurn],
        primary: GateDecision,
    ) -> GateDecision:
        fallback = await self._fallback.evaluate(text, history=history)
        return fallback.model_copy(update={"attempts": [primary, fallback]})

    async def evaluate(
        self, text: str, *, history: Sequence[ConversationTurn] = ()
    ) -> GateDecision:
        started = perf_counter()
        try:
            score = await self._client.score(text)
        except Exception as exc:
            latency_ms = int((perf_counter() - started) * 1000)
            logger.warning("scope_classifier_failed falling_back", exc_info=True)
            primary = GateDecision(
                name=self.name,
                verdict=GateVerdict.REJECT,
                passed=False,
                response="The scope classifier failed.",
                reason="Primary scope classifier execution failed; the fallback gate was used.",
                provider="model",
                model=self._client.model,
                confidence=1.0,
                latency_ms=latency_ms,
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return await self._defer(text, history, primary)

        latency_ms = int((perf_counter() - started) * 1000)
        probability = score.in_scope_probability

        # Is this request anything like what the classifier was trained on? The
        # grey zone cannot answer that: requests on unfamiliar topics do not land
        # near the threshold, they land confidently on one side of it. Measured on
        # production-shaped logs, three of four false positives scored 0.89-0.98
        # and every one of them was among the farthest requests from the corpus.
        detector = self._ood
        ood = await self._ood_verdict(text)
        if detector is not None and ood is not None and ood.far and self._ood_enforce:
            primary = GateDecision(
                name=self.name,
                verdict=GateVerdict.REJECT,
                passed=False,
                response="The scope classifier has not seen requests like this one.",
                reason=(
                    f"Scope classifier scored {probability:.3f}, but the request sits at "
                    f"similarity {ood.similarity:.3f} to the training corpus "
                    f"(quantile {ood.quantile:.1%}, threshold {detector.threshold:.3f}); "
                    f"the fallback gate decided."
                ),
                provider="model",
                model=self._client.model,
                confidence=probability,
                latency_ms=int((perf_counter() - started) * 1000),
            )
            return await self._defer(text, history, primary)

        if probability >= self._grey_high:
            return self._decision(passed=True, score=probability, latency_ms=latency_ms, ood=ood)
        if probability <= self._grey_low:
            if not self._confirm_reject:
                return self._decision(
                    passed=False, score=probability, latency_ms=latency_ms, ood=ood
                )
            # A low score is exactly where the missing dialogue hurts: "а что
            # рядом?" and "а второй?" score like fragments. The encoder never
            # rejects alone — the scoper reads the recent turns and decides.
            primary = GateDecision(
                name=self.name,
                verdict=GateVerdict.REJECT,
                passed=False,
                response="The scope classifier found nothing travel-related in this text alone.",
                reason=(
                    f"Scope classifier scored {probability:.3f}; a rejection is confirmed "
                    f"against the dialogue, so the fallback gate decided."
                ),
                provider="model",
                model=self._client.model,
                confidence=1.0 - probability,
                latency_ms=latency_ms,
            )
            return await self._defer(text, history, primary)

        # Inside the grey zone the classifier has no useful opinion. This is not a
        # failure, so the attempt is recorded as a success with its real score.
        primary = GateDecision(
            name=self.name,
            verdict=GateVerdict.REJECT,
            passed=False,
            response="The scope classifier was not confident enough.",
            reason=(
                f"Scope classifier scored {probability:.3f}, inside the grey zone "
                f"{self._grey_low:.2f}–{self._grey_high:.2f}; the fallback gate decided."
            ),
            provider="model",
            model=self._client.model,
            confidence=probability,
            latency_ms=latency_ms,
        )
        return await self._defer(text, history, primary)


class LLMCensorshipGate(GateEvaluator):
    """Censorship classification by an LLM, with the regex gate as a safety net.

    Deliberately the same shape as :class:`LLMScopeGate`: ``CENSOR_SYSTEM_PROMPT``
    asks for a bare ``safe``/``unsafe`` over an ordinary OpenAI chat completion, so
    any compatible backend serves it — vLLM, Ollama, a hosted API. Nothing here is
    tied to one model's chat template.

    Any failure (model down, timeout, unparseable answer) falls back to the
    deterministic gate. Unlike the scope gate, the stakes are asymmetric: this one
    also judges the model's ANSWER, so a dead classifier must not turn into a
    blanket allow.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str | None = None,
        fallback: GateEvaluator | None = None,
    ) -> None:
        self.name = GateName.CENSORSHIP
        self._llm = llm
        self._model = model
        self._fallback = fallback or RuleBasedCensorshipGate()

    async def evaluate(
        self, text: str, *, history: Sequence[ConversationTurn] = ()
    ) -> GateDecision:
        started = perf_counter()
        try:
            response = await self._llm.generate(
                LLMRequest(
                    messages=_censor_messages(text),
                    model=self._model,
                    temperature=0.0,
                    max_tokens=_CENSOR_MAX_TOKENS,
                )
            )
        except Exception as exc:
            primary_latency_ms = int((perf_counter() - started) * 1000)
            logger.warning("censorship_gate_llm_failed falling_back_to_rules", exc_info=True)
            primary = GateDecision(
                name=self.name,
                verdict=GateVerdict.REJECT,
                passed=False,
                response="The model-backed safety check failed.",
                reason="Primary censorship classifier execution failed; rules fallback was used.",
                provider="model",
                model=self._model,
                confidence=1.0,
                latency_ms=primary_latency_ms,
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            fallback = await self._fallback.evaluate(text, history=history)
            return fallback.model_copy(update={"attempts": [primary, fallback]})

        passed = parse_censor_verdict(response.content)
        if passed is None:
            primary_latency_ms = int((perf_counter() - started) * 1000)
            logger.warning(
                "censorship_gate_unparseable falling_back_to_rules content=%r",
                response.content[:200],
            )
            primary = GateDecision(
                name=self.name,
                verdict=GateVerdict.REJECT,
                passed=False,
                response="The model-backed safety result could not be parsed.",
                reason="Primary censorship classifier returned an unparseable verdict;"
                "rules fallback was used.",
                provider="model",
                model=response.model,
                confidence=1.0,
                latency_ms=primary_latency_ms,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                success=False,
                error_type="UnparseableGateResponse",
                error_message=(response.content or "")[:1000],
            )
            fallback = await self._fallback.evaluate(text, history=history)
            return fallback.model_copy(update={"attempts": [primary, fallback]})

        return GateDecision(
            name=self.name,
            verdict=GateVerdict.ALLOW if passed else GateVerdict.REJECT,
            passed=passed,
            response=(
                "No prohibited content was detected."
                if passed
                else "I cannot help with that request."
            ),
            reason=f"Censorship classifier answered {'safe' if passed else 'unsafe'}.",
            provider="model",
            model=response.model,
            # A bare safe/unsafe carries no probability; the verdict is discrete.
            confidence=1.0,
            matched_rules=[],
            latency_ms=int((perf_counter() - started) * 1000),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )


class GuardianCensorshipGate(GateEvaluator):
    """Censorship by IBM Granite Guardian, served by vLLM.

    Guardian is not steered by a prompt the way a general model is: its own chat
    template wraps the message in a safety instruction and it replies ``Yes``
    (the risk is present) or ``No`` (it is not). So this sends the text as a bare
    user turn — a system prompt of ours would fight the template — and reads the
    answer with :func:`parse_guardian_verdict`, which inverts it back into the
    pass/fail the rest of the pipeline expects.

    The call itself is an ordinary OpenAI chat completion. vLLM applies the
    template's default risk (``harm``) when no ``guardian_config`` is supplied,
    which is what keeps this free of vendor-specific request fields. Verified
    against granite-guardian-3.1-2b: a hospital search answers ``No``, weapon
    instructions answer ``Yes``.

    Failures fall back to the regex gate, same as every other model gate.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str | None = None,
        fallback: GateEvaluator | None = None,
    ) -> None:
        self.name = GateName.CENSORSHIP
        self._llm = llm
        self._model = model
        self._fallback = fallback or RuleBasedCensorshipGate()

    async def evaluate(
        self, text: str, *, history: Sequence[ConversationTurn] = ()
    ) -> GateDecision:
        started = perf_counter()
        try:
            response = await self._llm.generate(
                LLMRequest(
                    # No system turn: Guardian's template supplies the instruction.
                    messages=[LLMMessage(role="user", content=text)],
                    model=self._model,
                    temperature=0.0,
                    max_tokens=_CENSOR_MAX_TOKENS,
                )
            )
        except Exception as exc:
            primary_latency_ms = int((perf_counter() - started) * 1000)
            logger.warning("guardian_gate_failed falling_back_to_rules", exc_info=True)
            primary = GateDecision(
                name=self.name,
                verdict=GateVerdict.REJECT,
                passed=False,
                response="The Guardian safety check failed.",
                reason="Guardian execution failed; rules fallback was used.",
                provider="model",
                model=self._model,
                confidence=1.0,
                latency_ms=primary_latency_ms,
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            fallback = await self._fallback.evaluate(text, history=history)
            return fallback.model_copy(update={"attempts": [primary, fallback]})

        passed = parse_guardian_verdict(response.content)
        if passed is None:
            primary_latency_ms = int((perf_counter() - started) * 1000)
            logger.warning(
                "guardian_gate_unparseable falling_back_to_rules content=%r",
                response.content[:200],
            )
            primary = GateDecision(
                name=self.name,
                verdict=GateVerdict.REJECT,
                passed=False,
                response="The Guardian safety result could not be parsed.",
                reason="Guardian returned an unparseable verdict; rules fallback was used.",
                provider="model",
                model=response.model,
                confidence=1.0,
                latency_ms=primary_latency_ms,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                success=False,
                error_type="UnparseableGateResponse",
                error_message=(response.content or "")[:1000],
            )
            fallback = await self._fallback.evaluate(text, history=history)
            return fallback.model_copy(update={"attempts": [primary, fallback]})

        return GateDecision(
            name=self.name,
            verdict=GateVerdict.ALLOW if passed else GateVerdict.REJECT,
            passed=passed,
            response=(
                "No prohibited content was detected."
                if passed
                else "I cannot help with that request."
            ),
            reason=f"Granite Guardian answered {'no' if passed else 'yes'} to the harm risk.",
            provider="model",
            model=response.model,
            confidence=1.0,
            matched_rules=[] if passed else ["harm"],
            latency_ms=int((perf_counter() - started) * 1000),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )


def get_preflight_gates() -> tuple[GateEvaluator, GateEvaluator]:
    """Factory seam for swapping regex gates with model-backed adapters later."""
    return RuleBasedScopeGate(), RuleBasedCensorshipGate()


async def evaluate_preflight_gates(
    text: str,
    *,
    scope_gate: GateEvaluator | None = None,
    censorship_gate: GateEvaluator | None = None,
    history: Sequence[ConversationTurn] = (),
) -> GateResults:
    """Evaluate both input gates concurrently with independent failure timers."""
    if scope_gate is None or censorship_gate is None:
        default_scope, default_censorship = get_preflight_gates()
        scope_gate = scope_gate or default_scope
        censorship_gate = censorship_gate or default_censorship

    async def _measured(
        gate: GateEvaluator,
        *,
        gate_history: Sequence[ConversationTurn] = (),
    ) -> GateDecision:
        started = perf_counter()
        try:
            return await gate.evaluate(text, history=gate_history)
        except Exception as exc:
            latency_ms = int((perf_counter() - started) * 1000)
            logger.exception("input_gate_failed gate=%s", gate.name.value)
            return GateDecision(
                name=gate.name,
                verdict=GateVerdict.REJECT,
                passed=False,
                response="The safety check failed; the request cannot continue.",
                reason="Gate execution failed.",
                provider="model",
                model=getattr(gate, "model_name", None) or getattr(gate, "_model", None),
                confidence=1.0,
                latency_ms=latency_ms,
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    scope_result, censorship_result = await asyncio.gather(
        _measured(scope_gate, gate_history=history),
        _measured(censorship_gate),
    )

    if scope_result.name is not GateName.SCOPE:
        raise ValueError("scope_gate returned a decision with the wrong name")
    if censorship_result.name is not GateName.CENSORSHIP:
        raise ValueError("censorship_gate returned a decision with the wrong name")
    return GateResults(scope=scope_result, censorship=censorship_result)


def resolve_censorship_gate(override: GateEvaluator | None = None) -> GateEvaluator:
    """The censorship gate to use: the injected one, or a fresh rule-based gate.

    The injected gate (e.g. the loaded model) is reused across input and output
    checks, so the model is loaded once and inferred twice per request.
    """
    return override if override is not None else RuleBasedCensorshipGate()


async def evaluate_output_censorship(
    answer: str, *, censorship_gate: GateEvaluator | None = None
) -> GateDecision:
    """Run the censorship gate on the MODEL's answer (post-generation check).

    Returns a decision tagged :attr:`GateName.CENSORSHIP_OUTPUT` so it is
    distinguishable from the input-stage censorship check in logs and responses.
    """
    started = perf_counter()
    try:
        decision = await resolve_censorship_gate(censorship_gate).evaluate(answer)
    except Exception as exc:
        logger.exception("output_censorship_gate_failed")
        return GateDecision(
            name=GateName.CENSORSHIP_OUTPUT,
            verdict=GateVerdict.REJECT,
            passed=False,
            response="The safety check failed; the response cannot be returned.",
            reason="Output censorship gate execution failed.",
            provider="model",
            model=None,
            confidence=1.0,
            latency_ms=int((perf_counter() - started) * 1000),
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    return decision.model_copy(update={"name": GateName.CENSORSHIP_OUTPUT})

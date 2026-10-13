# Shadow Lab Decision Rules & Reason Codes

## 1. Physical Coach Consensus & The $\ge 3$ Distinct Coach Rule

### Fundamental Invariant
> Repeated frames observing the same physical coach segment count as **exactly ONE** coach observation. 50 frames of Coach C01 = 1 vote.

### Audited Numeric Confirmation Policy
- **`PAIR_CONFIRMED`:** A bidirectional numeric candidate pair (e.g. `[12919, 12920]`) is marked `PAIR_CONFIRMED` if and only if **both individual candidates** reach $\ge 3$ distinct physical coaches ($\ge 3$ distinct `coach_segment_id` occurrences).
- **`PAIR_PROVISIONAL`:** If one candidate reaches $\ge 3$ distinct coaches while the opposite direction candidate is observed on only 1 or 2 coaches (e.g., historical Video 8 where `12920` had 5 coaches but `12919` had only 2), the pair is classified as `PAIR_PROVISIONAL`.
- **`PAIR_UNCONFIRMED`:** If no candidate reaches 3 distinct physical coaches, the pair is unconfirmed.

---

## 2. Decision Pathways

### Pathway A: Number-First (Reliable 2-Number Board)
1. Extract stable candidate pair from visual observations.
2. Intersect candidates with station timetable within the early/late operating window (`-15m` to `+90m`).
3. If exactly one candidate is operational and corroborated by NTES, select that active service number.
4. If both candidates remain plausible (e.g. turnaround delay or timetable bidirectional symmetry), abstain with status `AMBIGUOUS`. Never force an arbitrary guess.

### Pathway B: Name-First (Complex 4-Number or 6-Number Boards)
1. Do not guess active service numbers from digit positions or layout.
2. Extract consensus train name across coaches (e.g. `MALWA EXPRESS`).
3. Look up train family in site timetable and alias dictionary.
4. Intersect the visual number set with the scheduled family service set.
5. Select active number only when corroborated by station time window and NTES.

### Pathway C: Non-Numeric Fallback (Train Name + Endpoints)
1. When route boards display no 5-digit service numbers (e.g. `TAJ EXPRESS`, `LUCKNOW MAIL`), stabilize name and endpoints across coaches.
2. Match normalized name against site timetable.
3. Check platform context and scheduled arrival/departure.
4. Emit `PROVISIONAL` or `CONFIRMED` with explicit reason code `VISUAL_BOARD_NOT_NUMBERED`.

---

## 3. Negative-Class Filtering
The pipeline actively rejects non-service tokens:
- **Coach Serial Numbers:** 6-digit coach IDs (e.g. `217152`, `217152/C`) are rejected.
- **Berth / Stencil Markings:** Class codes and seat ranges (e.g. `S7`, `43-48`, `B1 1-72`) are rejected.
- **Fragments:** 3-digit or 4-digit numbers (e.g. `1292`) are never autocompleted without visual evidence.

---

## 4. Conflict & Abstention Gates
- **`RC_VISUAL_NTES_CONFLICT`:** If visual numbers directly contradict the NTES board (e.g. visual `[12919, 12920]` vs NTES reporting `12001`), automatic confirmation is suspended and status is set to `CONFLICT`.
- **`RC_PLATFORM_AMBIGUOUS_ISLAND`:** When an island camera observes tracks without individual platform discrimination, platform certainty is never fabricated.
- **`RC_NOT_SERVICE_EVENT`:** EMU/MEMU rake movements (numbers starting with 6) are gated separately to prevent overwriting express passenger identities.

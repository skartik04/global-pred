# Dr. Raj Feedback Analysis — Drug Prediction Pipeline

**Reviewer:** Dr. Raj
**Sample:** 20 patients × 3 visits = 60 reviews
**Sample composition:** 10 poly_wrong, 5 mono_wrong, 3 poly_correct, 2 mono_correct

---

## 1. Results (all 20 patients sampled based on V1 outcome)


| Patient                       | Category     | Dr. Raj | Comment                                                                                         |
| ----------------------------- | ------------ | ------- | ----------------------------------------------------------------------------------------------- |
| 262_Mutyaba Derrick           | poly_wrong   | ✅       | —                                                                                               |
| 64_Owinyi Golden              | poly_wrong   | ✅       | —                                                                                               |
| 10_Muduku Matthew             | poly_wrong   | ✅       | *"Would have done what the model recommended, not the neurologist"*                             |
| 288_Agonzibwa Trevor          | poly_wrong   | ✅       | *"CBZ input is wrong"*                                                                          |
| 95_Lomakol Ives Zane          | poly_wrong   | ✅       | *"Continuation of CBZ monotherapy with dose increase is valid"*                                 |
| 90_Odongo Moses               | poly_wrong   | ✅       | —                                                                                               |
| 85_Mirembe Mercy              | poly_wrong   | ✅       | *"Adding a second drug before maximizing is not warranted"*                                     |
| 67_Ssenyonjo Waswa            | poly_wrong   | ✅       | *"Optimizing monotherapy"*                                                                      |
| 320_Nakibirango Calvin        | poly_wrong   | ✅       | *"LLM suggestions appropriate even though they don't reflect neurologist decision"*             |
| 58_Sekyeru Jeremiah           | poly_wrong   | ✅       | *"LLM's reasoning given the clinical input is correct"*                                         |
| 317_Nantume Shabila           | mono_wrong   | ✅       | *"Agree with both the LLM and neurologist plans"*                                               |
| 39_Najjemba Christine         | mono_wrong   | ❌       | *"VPA could be considered. Seizure type unknown — broad spectrum needed. VPA is penalized"*     |
| 260_Mukisa Elizabeth          | mono_wrong   | ✅       | *"VPA is appropriate as suggested by LLM. CBZ also appropriate"*                                |
| 9_Nalukwaago Patience         | mono_wrong   | ✅       | *"CBZ could be continued. LLM identifying seizures as generalized is not completely incorrect"* |
| 8_Tisma Natabi                | mono_wrong   | ✅       | —                                                                                               |
| 74_Mukisa Shalom              | poly_correct | ✅       | *"If pt is seizure free on dual treatment that should be continued"*                            |
| 131_Muwanguzi Blessed Serugga | poly_correct | ❌       | *"If pt is seizure free on 2 drugs, that needs to be continued"*                                |
| 100_Otim Fortunate            | poly_correct | ✅       | *"If pt is on 2 drugs — continue. Option 2 and 3 appropriate"*                                  |
| 227_Muluwaya Arnold           | mono_correct | ✅       | —                                                                                               |
| 272_Ogol Jerry                | mono_correct | ✅       | —                                                                                               |


**18/20 = 90% agree** on qualifying visits. 2 Nos.


| Category     | Total reviews | Agree | Disagree | Agreement   |
| ------------ | ------------- | ----- | -------- | ----------- |
| poly_wrong   | 10            | 10    | 0        | **100.0%**  |
| mono_wrong   | 5             | 4     | 1        | **80.0%**   |
| poly_correct | 3             | 2     | 1        | **67.0%**   |
| mono_correct | 2             | 2     | 0        | **100.0%**  |
| **Total**    | **20**        | **18**| **2**    | **90.0%**   |


---

## 2. Agreement by Category — All 60 Reviews (Per-Visit Category)

Each of the 60 reviews (20 patients × 3 visits) is categorised independently based on that visit's own grading outcome, not the V1 sampling category.

| Category     | Total reviews | Agree | Disagree | Agreement  |
| ------------ | ------------- | ----- | -------- | ---------- |
| poly_wrong   | 26            | 21    | 5        | **80.8%**  |
| poly_correct | 13            | 12    | 1        | **92.3%**  |
| mono_wrong   | 6             | 5     | 1        | **83.3%**  |
| mono_correct | 15            | 15    | 0        | **100.0%** |
| **Total**    | **60**        | **53**| **7**    | **88.3%**  |


---

## 3. Agreement by Visit


| Visit   | Agree             | n   |
| ------- | ----------------- | --- |
| Visit 1 | 18/20 = **90.0%** | 20  |
| Visit 2 | 18/20 = **90.0%** | 20  |
| Visit 3 | 17/20 = **85.0%** | 20  |


## 4. The 7 Disagreements (No) — Full Detail


| Patient                           | Visit | Category     | Dr. Raj's comment                                                                                                                                                                                                       |
| --------------------------------- | ----- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **64_Owinyi Golden**              | V2    | poly_wrong   | *"agree with the reasoning but not the prediction. I will continue the neurologist's prediction."*                                                                                                                      |
| **64_Owinyi Golden**              | V3    | poly_wrong   | *"It is not recognizing that polytherapy is needed."*                                                                                                                                                                   |
| **85_Mirembe Mercy**              | V3    | poly_wrong   | *"Now that pt is continuing to have seizures on 2 medications, agree with neurologist plan that a new combo of dual ASM is needed."*                                                                                    |
| **39_Najjemba Christine**         | V1    | mono_wrong   | *"the girl child is 3 yrs of age and VPA could be considered. Also, given seizure type is unknown, broad spectrum is needed. The LLM recognizes it but still prescribes CBZ, as VPA is penalized. LEV is appropriate."* |
| **131_Muwanguzi Blessed Serugga** | V1    | poly_correct | *"If pt is seizure free on 2 drugs, that needs to be continued."*                                                                                                                                                       |
| **100_Otim Fortunate**            | V2    | poly_wrong   | *"medically refractory on 3 drugs. so there are several optimal choices including option 2."*                                                                                                                           |
| **100_Otim Fortunate**            | V3    | poly_wrong   | *"Agree with neurologist. Option 2 suggested by LLM is appropriate."*                                                                                                                                                   |


---









&nbsp;

&nbsp;

&nbsp;
&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
&nbsp;

&nbsp;

&nbsp;





&nbsp;

&nbsp;

&nbsp;
&nbsp;

&nbsp;

&nbsp;
&nbsp;

&nbsp;

&nbsp;





&nbsp;

&nbsp;

&nbsp;




&nbsp;

&nbsp;

&nbsp;
















## 6. Issue Categories — Patient Lists

---

### 6A. Dr. Raj Disagrees with Both Model AND Neurologist

*(No vote in poly_correct or mono_correct — GT matched but still wrong)*

These are the most important cases to debug: the model predicted what the neurologist prescribed, yet the clinical expert disagrees with both.


| Patient                           | Visit | What happened                                                                                                                      |
| --------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **131_Muwanguzi Blessed Serugga** | V1    | Patient is seizure-free on CBZ + LEV (dual therapy). Model pushes monotherapy optimization. Dr. Raj says: continue what's working. |


---

### 6B. Dr. Raj Explicitly Prefers Model Over Neurologist

*(Yes vote in poly_wrong or mono_wrong — model didn't match GT but Dr. Raj agrees with model)*

These cases suggest the grading metric **underestimates** model quality. The model may be clinically more correct than the accuracy score shows.


| Patient                    | Visit | Dr. Raj's comment                                                                                                                                       |
| -------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **10_Muduku Matthew**      | V1    | *"agree with the model. Would have done what is recommended by the model and not the neurologist."*                                                     |
| **67_Ssenyonjo Waswa**     | V3    | *"Option 2 again is ideal. CBZ was worsening drop attack seizures. During 3rd visit the neurologist recognized this as well."*                          |
| **320_Nakibirango Calvin** | V1    | *"monotherapy is preferred. LLM suggestions are appropriate, even though they do not reflect the neurologist decision."*                                |
| **320_Nakibirango Calvin** | V2    | *"Agree with starting VPA given pt's history as suggested by LLM."*                                                                                     |
| **320_Nakibirango Calvin** | V3    | *"The LLM's prediction was finally adopted by the neurologist as well in the last visit."*                                                              |
| **90_Odongo Moses**        | V2    | *"Monotherapy should always be considered first. Option 1 and 2 chosen by LLM is appropriate. Instead of option 3, I would prefer neurologist option."* |
| **90_Odongo Moses**        | V3    | *"Agree with LLM reasoning, but also appreciate the neurologist's plan to continue drugs that is working."*                                             |
| **58_Sekyeru Jeremiah**    | V1    | *"LLM's reasoning given the clinical input is correct."*                                                                                                |
| **317_Nantume Shabila**    | V1    | *"agree with both the LLM and neurologist plans."*                                                                                                      |
| **317_Nantume Shabila**    | V3    | *"would switch to another monotherapy as the LLM suggested."*                                                                                           |
| **260_Mukisa Elizabeth**   | V1    | *"VPA is an appropriate choice as suggested by LLM. BUT also CBZ is appropriate."*                                                                      |
| **9_Nalukwaago Patience**  | V1–V3 | *"Choices provided by LLM are all valid. Reasoning is appropriate."*                                                                                    |


---

### 6C. Input Errors / Leakage Flagged

*(Model given wrong medication history — pipeline bug, not model failure)*


| Patient                  | Visit | Error                                                                                                                                                                        | Dr. Raj's comment                                                                                                                       |
| ------------------------ | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **262_Mutyaba Derrick**  | V2    | PHB shows as active but was stopped at V1. Extraction hallucinated PHB prescription that never existed.                                                                      | *"Input Phenobarbital is not currently on. It was stopped last visit."*                                                                 |
| **262_Mutyaba Derrick**  | V3    | PHB still shows as active across all 3 visits. Raw CSV shows VPA only from V2 onward.                                                                                        | *"Why is the model thinking that the Phenobarbital was continued. The input is incorrect."*                                             |
| **288_Agonzibwa Trevor** | V1    | CBZ backfilled as "on arrival" by reconciliation. Patient was drug-naive at V1 (`Medication status: Not on Medication` in raw CSV). CBZ was started at V1, not pre-existing. | *"it is unclear why the model sees that the pt was on carbamazepine. Pt was started during visit 1. information leakage is occurring."* |
| **288_Agonzibwa Trevor** | V3    | CBZ→PHB drug switch at V3 not captured. Model sees VPA+CBZ when reality is VPA+PHB.                                                                                          | *"PHB and valproate are the two drugs pt is on. This is not being recognized. It still thinks it's on VPA, CBZ."*                       |
| **58_Sekyeru Jeremiah**  | V1    | 4 ASMs prescribed simultaneously — suggests prior medication history missing entirely.                                                                                       | *"unclear if pt was on any medication prior to first visit. 4 ASM were given at the same time."*                                        |


**Root causes confirmed:**

- `262`: `extract_phenos.py` hallucinated PHB; `reconc_all.py` carried it forward
- `288 V1`: `fix_drug_backfill.py` backfilled CBZ to V1 as "on arrival"
- `288 V3`: `reconc_all.py` missed CBZ→PHB switch

---

### 6D. Seizure-Free Status Ignored

*(Model doesn't know whether current regimen is working)*

This is the **most frequent issue** (9 comments). The model has no `SeizureControl` feature — it always reasons from drug history alone and defaults to monotherapy optimization, even when the patient is stable and seizure-free.


| Patient                           | Visit | Dr. Raj's comment                                                                                                    |
| --------------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------- |
| **131_Muwanguzi Blessed Serugga** | V1    | *"If pt is seizure free on 2 drugs, that needs to be continued."*                                                    |
| **131_Muwanguzi Blessed Serugga** | V3    | *"option 3 (polytherapy needs to be considered) as pt is seizure free."*                                             |
| **74_Mukisa Shalom**              | V1    | *"agree with the dual treatment continuation. If pt is seizure free on dual treatment that should be continued."*    |
| **74_Mukisa Shalom**              | V2    | *"LLM is optimizing monotherapy before polytherapy. But if pt is seizure on polytherapy, that should be continued."* |
| **95_Lomakol Ives Zane**          | V3    | *"unclear from the note, if the patient continues to have seizures or remain seizure free."*                         |
| **227_Muluwaya Arnold**           | V2    | *"if pt is seizure free, no need to change or provide additional options."*                                          |
| **272_Ogol Jerry**                | V2    | *"if pt is seizure free on current regimen, no need to change."*                                                     |
| **317_Nantume Shabila**           | V2    | *"now that LLM sees that with VPA, pt is seizure free it's recommending continuation."* (positive)                   |


**Proposed fix:** Add `SeizureControl` as an extracted feature: `seizure_free / reduced / unchanged / worse`. One field, highest clinical leverage.

---

### 6E. Monotherapy Push Too Aggressive

*(Model fails to escalate to polytherapy when clearly needed)*


| Patient              | Visit | Dr. Raj's comment                                                                                                                    |
| -------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **64_Owinyi Golden** | V2    | *"agree with the reasoning but not the prediction. I will continue the neurologist's prediction."*                                   |
| **64_Owinyi Golden** | V3    | *"It is not recognizing that polytherapy is needed."*                                                                                |
| **85_Mirembe Mercy** | V3    | *"Now that pt is continuing to have seizures on 2 medications, agree with neurologist plan that a new combo of dual ASM is needed."* |


Both cases involve persistent seizures on current therapy — the model keeps proposing monotherapy adjustment instead of recognizing treatment failure and escalating. Directly linked to issue 6D (no seizure control signal).

---

### 6F. Refractory Cases Not Handled

*(Model doesn't escalate options enough for drug-resistant patients)*


| Patient                 | Visit | Dr. Raj's comment                                                                                |
| ----------------------- | ----- | ------------------------------------------------------------------------------------------------ |
| **58_Sekyeru Jeremiah** | V2    | *"these are drug refractory patients and likely difficulty to treat and most drugs will fail."*  |
| **58_Sekyeru Jeremiah** | V3    | *"drug resistant. could continue the neurologist suggestion or also option 2 suggested by LLM."* |
| **100_Otim Fortunate**  | V2    | *"medically refractory on 3 drugs. so there are several optimal choices including option 2."*    |
| **100_Otim Fortunate**  | V3    | *"Agree with neurologist. Option 2 suggested by LLM is appropriate."*                           |


---

### 6G. Between-Visit Drug Changes Missing


| Patient                  | Visit | Dr. Raj's comment                                                                                            |
| ------------------------ | ----- | ------------------------------------------------------------------------------------------------------------ |
| **288_Agonzibwa Trevor** | V2    | *"valproate was added in another visit and now pt is on 2 drugs. this was done in a visit between 1 and 2."* |


Pipeline captures only 3 timepoints (0, 6, 12 months). Drug changes in between-visit clinic appointments are invisible to the model.

---

### 6H. Drug Availability Not Modelled


| Patient             | Visit | Dr. Raj's comment                                                                                                                                                           |
| ------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **90_Odongo Moses** | V3    | *"VPA is more commonly available than LEV and provided by the national formulary. One option: where drug availability input is provided by the provider before the visit."* |


VPA > LEV in the Ugandan national formulary. The model correctly knows this at the prompt level but may not weight it strongly enough. Dr. Raj's suggestion: add formulary availability as a structured input field.

---

## 7. Summary: What to Fix


| Priority  | Issue                                         | Fix                                                                                                     |
| --------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 🔴 High   | Seizure control outcome not extracted         | Add `SeizureControl` feature: free / reduced / unchanged / worse                                        |
| 🔴 High   | PHB hallucinated for 262                      | Fix `extract_phenos.py` extraction for 262_Mutyaba                                                      |
| 🔴 High   | CBZ leakage at V1 for 288                     | Fix `fix_drug_backfill.py` — don't backfill as "on arrival" when `Medication status: Not on Medication` |
| 🔴 High   | CBZ→PHB switch not captured for 288 V3        | Fix `reconc_all.py` drug transition logic                                                               |
| 🟡 Medium | Monotherapy prompt too aggressive             | Add guidance for refractory cases and seizure-free continuation                                         |
| 🟡 Medium | VPA over-penalized for ambiguous seizure type | Soften prompt anchor when seizure type unclear                                                          |
| 🟢 Low    | Drug availability not in input                | Add formulary/availability field as optional provider input                                             |



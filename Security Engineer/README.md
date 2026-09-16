\# Security Engineer



\*\*Team Member:\*\* Iswarya

\*\*Role:\*\* Risk/Security Engineer



\## 1. Overview



The Security Engineer module provides the risk assessment and security decision layer for the AI voice deepfake detection system.



It combines multiple security signals, performs contextual risk analysis, monitors repeated suspicious activity, generates alerts and recommendations, and produces an explainable security decision.



\## 2. Responsibilities



\* Risk engine

\* Risk scoring and classification

\* Contextual risk analysis

\* Correlation and escalation analysis

\* Security policy decisions

\* Risk monitoring

\* Alerts and recommendations

\* Audit logging

\* File/upload security validation

\* Security testing



\## 3. System Architecture



```text

AI / Security Signals

&#x20;       |

&#x20;       v

Signal Validation

&#x20;       |

&#x20;       v

Speaker Signal Normalization

&#x20;       |

&#x20;       v

Weighted Risk Fusion

&#x20;       |

&#x20;       v

Correlation Analysis

&#x20;       |

&#x20;       v

Risk Score + Risk Level

&#x20;       |

&#x20;       v

Contextual Risk Analysis

&#x20;       |

&#x20;       v

Risk Monitoring / Escalation

&#x20;       |

&#x20;       v

Security Policy

&#x20;       |

&#x20;       v

Alert + Recommendation

&#x20;       |

&#x20;       v

Audit Logging

```



\## 4. Risk Engine



The main risk engine is implemented in:



```text

backend/services/risk\_engine.py

```



The engine accepts the following security signals:



\* AI-generated voice probability

\* Speaker similarity

\* Prosody anomaly

\* Caller anomaly

\* Transaction risk



All numerical signals are validated to ensure that they are within the range \*\*0-100\*\*.



\### Risk Weights



| Signal                         |   Weight |

| ------------------------------ | -------: |

| AI-generated voice probability |      40% |

| Speaker anomaly                |      20% |

| Prosody anomaly                |      10% |

| Caller anomaly                 |      15% |

| Transaction risk               |      15% |

| \*\*Total\*\*                      | \*\*100%\*\* |



Speaker similarity is converted into speaker anomaly because high speaker similarity represents lower risk:



```text

speaker\_anomaly = 100 - speaker\_similarity

```



The weighted signals are combined to calculate the base risk score.



\## 5. Risk Classification



The calculated risk score is classified into four levels:



| Risk Score | Risk Level |

| ---------: | ---------- |

|       0-29 | LOW        |

|      30-59 | MEDIUM     |

|      60-79 | HIGH       |

|     80-100 | CRITICAL   |



The final score is constrained to the range 0-100.



\## 6. Correlation and Escalation



The engine does not rely only on individual signals.



It also checks combinations of suspicious signals, including:



\* High AI probability + speaker mismatch

\* High AI probability + caller anomaly

\* Caller anomaly + high transaction risk

\* High AI probability + high transaction risk

\* Multiple simultaneous high-risk indicators

\* AI impersonation pattern involving caller and transaction risk



These combinations can add an escalation bonus to the weighted risk score.



This allows the system to identify situations where several independent indicators occur together.



\## 7. Explainable Risk Assessment



The risk engine returns information explaining why a call received its risk score.



The result includes:



\* Risk score

\* Risk level

\* Confidence

\* Original signal values

\* Signal contributions

\* Individual risk factors

\* Correlation factors

\* Escalation bonus

\* Alert

\* Recommendation

\* Security decision



This makes the risk assessment explainable rather than returning only a numerical score.



\## 8. Security Decisions



The engine maps risk levels to security actions.



\### LOW



\*\*Decision:\*\*



```text

ALLOW\_WITH\_NORMAL\_VERIFICATION

```



The system continues normal verification procedures.



\### MEDIUM



\*\*Decision:\*\*



```text

VERIFY

```



Additional caller verification is recommended and the request should be monitored.



\### HIGH



\*\*Decision:\*\*



```text

STEP\_UP\_AUTHENTICATION

```



The caller should be independently verified and additional authentication should be required.



\### CRITICAL



\*\*Decision:\*\*



```text

BLOCK\_AND\_ESCALATE

```



The requested transaction should not be authorized. Independent callback, multi-factor authentication, and escalation to a supervisor or security team are recommended.



\## 9. Contextual Risk Analysis



Contextual risk is implemented in:



```text

backend/services/contextual\_risk.py

```



The contextual layer considers additional circumstances surrounding the call, including:



\* Whether the caller is known

\* Whether the request is urgent

\* Whether the transaction is high-value

\* Whether it is a first-time request



These contextual conditions can increase the overall risk assessment.



The contextual layer also supports security escalation when multiple contextual risk factors occur together.



\## 10. Risk Monitoring



Repeated suspicious activity is monitored using:



```text

backend/services/risk\_monitor.py

```



The monitoring component tracks risk activity within a defined monitoring window.



Repeated high-risk activity can trigger an escalation override and result in:



```text

BLOCK\_AND\_ESCALATE

```



with immediate security priority.



\## 11. Security Policy



Security policy handling is implemented in:



```text

backend/services/security\_policy.py

```



The policy layer converts the risk assessment and contextual information into an appropriate security response.



This separates risk calculation from the final security policy decision.



\## 12. Alerts and Recommendations



Alert and response handling is implemented in:



```text

backend/services/alert\_action.py

```



The system generates security alerts and recommendations based on the assessed risk level.



Examples include:



\* Potential AI voice impersonation detected

\* Suspicious voice call detected

\* Additional caller verification required

\* Additional authentication required

\* Transaction should be blocked and escalated



\## 13. Audit Logging



Security events are recorded through:



```text

backend/services/audit\_logger.py

```



Audit events contain information such as:



\* Event ID

\* UTC timestamp

\* Risk information

\* Security decision

\* Relevant security information



The generated security event log is stored in:



```text

backend/logs/security\_events.jsonl

```



\## 14. File/Upload Security



File security validation is implemented in:



```text

backend/services/file\_security.py

```



The module validates uploaded files before they enter the AI/security pipeline.



The following five security checks are implemented:



\### 1. Invalid File Type Detection



Only supported audio file extensions are accepted.



Supported file types include:



```text

.wav

.mp3

.m4a

.flac

.ogg

```



\### 2. Suspicious Upload Detection



The module detects suspicious uploads including:



\* Missing file extensions

\* Empty uploads

\* Unsupported file types



\### 3. Missing File Detection



The system verifies that:



\* The uploaded path exists

\* The uploaded path represents a regular file



\### 4. Oversized File Detection



Files larger than the configured maximum size of \*\*25 MB\*\* are rejected.



\### 5. Unsafe Filename Detection



The system checks filenames for unsafe characters and path-traversal-related patterns.



\### Combined Upload Validation



The combined validation function is:



```text

validate\_upload()

```



It performs all five security checks and returns the overall upload security status, failed checks, and recommendation.



For a valid upload:



```text

ALLOW\_UPLOAD

```



For an invalid or suspicious upload:



```text

REJECT\_UPLOAD

```



This provides a security validation layer before uploaded audio is processed by the remaining AI and security components.



\## 15. Testing



Security testing is implemented in:



```text

backend/tests/test\_risk\_security.py

```



The Risk/Security test suite covers:



1\. Low-risk assessment

2\. Medium-risk assessment

3\. High-risk assessment

4\. Critical-risk assessment

5\. Repeated-risk override

6\. Security override

7\. Caller/session isolation

8\. Invalid AI probability handling

9\. Invalid boolean input handling

10\. Audit event generation

11\. Final security decision consistency



\### File Security Tests



The file-security test suite is implemented in:



```text

backend/tests/test\_file\_security.py

```



It covers:



1\. Valid file validation

2\. Invalid file type

3\. Missing file

4\. Oversized file

5\. Unsafe filename

6\. Suspicious empty upload

7\. Suspicious unsupported upload

8\. Combined upload rejection



\### Complete Test Result



The complete Security Engineer test suite contains:



```text

11 Risk/Security tests

8 File Security tests

19 tests total

```



Test execution result:



```text

19 passed in 0.32s

```



All 19 Security Engineer tests passed successfully.



\## 16. Module Structure



```text

Security Engineer/

│

├── README.md

│

└── backend/

&#x20;   ├── logs/

&#x20;   │   └── security\_events.jsonl

&#x20;   │

&#x20;   ├── services/

&#x20;   │   ├── alert\_action.py

&#x20;   │   ├── audit\_logger.py

&#x20;   │   ├── contextual\_risk.py

&#x20;   │   ├── file\_security.py

&#x20;   │   ├── risk\_engine.py

&#x20;   │   ├── risk\_monitor.py

&#x20;   │   └── security\_policy.py

&#x20;   │

&#x20;   └── tests/

&#x20;       ├── test\_file\_security.py

&#x20;       └── test\_risk\_security.py

```



\## 17. Member 6 Deliverable Status



| Component                  | Status    |

| -------------------------- | --------- |

| Risk Engine                | Completed |

| Weighted Risk Scoring      | Completed |

| Risk Classification        | Completed |

| Correlation Analysis       | Completed |

| Contextual Risk            | Completed |

| Risk Monitoring            | Completed |

| Security Policy            | Completed |

| Alerts and Recommendations | Completed |

| Audit Logging              | Completed |

| File/Upload Security       | Completed |

| Security Testing           | Completed |

| 11 Risk/Security Tests     | Passed    |

| 8 File Security Tests      | Passed    |

| 19 Total Security Tests    | Passed    |



\*\*Member 6 Risk/Security module: Completed, tested, and documented.\*\*




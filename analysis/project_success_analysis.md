# AI Enablement and Project Outcomes Study Summary

## Study Design
- **Sample size:** 225 professionals working on AI-enabled projects.
- **Scale:** 5-point Likert (1 = strongly disagree, 5 = strongly agree).
- **Construct reliability:** Cronbach's alphas ranged from 0.81 to 0.91, supporting internal consistency for AI capability, team collaboration, project complexity, and project success scales.

## Descriptive Statistics
| Construct | Items | Mean | SD | Alpha |
|-----------|-------|------|----|-------|
| AI Capability | 5 | 3.79 | 0.65 | 0.84 |
| Team Collaboration | 5 | 3.58 | 0.72 | 0.88 |
| Project Complexity | 5 | 3.19 | 0.69 | 0.81 |
| Project Success | 6 | 3.89 | 0.71 | 0.91 |

Average ratings suggest respondents generally agreed that AI capabilities and collaboration were present, projects were moderately complex, and outcomes skewed positive.

## Zero-Order Correlations
- AI capability correlated strongly with team collaboration (*r* = .83) and modestly with project success (*r* = .34). Its link with project complexity was negligible and slightly negative (*r* = -.12).
- Team collaboration showed a moderate positive correlation with success (*r* = .42) and a small negative association with complexity (*r* = -.15).
- More complex projects tended to report lower success (*r* = -.25).

These patterns hint that collaboration may play a key role in translating AI investments into better outcomes while complexity can dampen success.

## Multiple Regression on Project Success
The model including AI capability, team collaboration, and project complexity explained 23.9% of the variance in project success (*F*(3, 221) = 23.11, *p* < .001).

| Predictor | B | SE | Beta | *t* | *p* |
|-----------|----|----|------|-----|-----|
| AI Capability | 0.21 | 0.08 | 0.21 | 2.63 | .009 |
| Team Collaboration | 0.36 | 0.07 | 0.36 | 5.14 | < .001 |
| Project Complexity | -0.18 | 0.07 | -0.18 | -2.57 | .011 |

- Greater team collaboration showed the largest standardized effect on success.
- AI capability retained a positive contribution even after controlling for collaboration and complexity.
- Higher complexity exerted a detrimental effect on outcomes.

## Mediation Analysis (PROCESS Model 4)
- **a-path (AI → Collaboration):** B = 0.91, *p* < .001.
- **b-path (Collaboration → Success):** B = 0.35, *p* < .001.
- **Direct effect (c′: AI → Success):** B = 0.18, *p* = .102 (ns).
- **Indirect effect:** 0.32, 95% bootstrap CI [0.15, 0.50].

Together, AI capability significantly enhances team collaboration, which in turn boosts success. Because the direct path becomes non-significant once collaboration is included, the findings support **full mediation**: collaboration is the key mechanism linking AI capability to project success.

## Moderation Analysis (PROCESS Model 1)
Testing whether project complexity alters the AI → success link yielded a non-significant interaction (*B* = 0.049, *p* = .606). Thus, complexity does **not** moderate the AI–success relationship in this dataset, although it still shows a main-effect drag on outcomes.

## Practical Implications
1. **Invest in collaborative practices.** Collaboration appears to be the dominant driver of success and the pathway through which AI capability delivers value.
2. **Manage complexity proactively.** Even without moderating the AI effect, higher complexity erodes success, highlighting the need for scope control and risk mitigation.
3. **Build AI readiness holistically.** Strengthening AI capabilities alongside collaboration practices offers the best opportunity for superior project results.


# policy/treatment_policy.py

from policy.policy_utils import total_reward
from policy.strategy_selector import DeltaNimStrategySelector

class TreatmentPolicy:
    def __init__(self, response_model, survival_model, action_space):
        self.response_model = response_model
        self.survival_model = survival_model
        self.selector = DeltaNimStrategySelector(action_space)

    def choose_action(self, state):
        candidates = self.selector.generate_moves(state)

        best_action = None
        best_score = -1e9

        for action, next_state in candidates:

            # ML predictions
            survival = self.survival_model.predict(next_state)
            response = self.response_model.predict_proba(next_state)[1]

            toxicity = next_state.get("toxicity", 0)
            progression = 1 - response

            score = total_reward(survival, toxicity, progression)

            if score > best_score:
                best_score = score
                best_action = {
                    "action": action,
                    "score": score,
                    "pred_survival": survival,
                    "response_prob": response
                }

        return best_action
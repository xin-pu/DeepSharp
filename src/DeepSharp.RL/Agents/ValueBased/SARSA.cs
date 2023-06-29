using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class SARSA : ValueAgent
    {
        public SARSA(Environ<Space, Space> env,
            float epsilon = 0.1f,
            float alpha = 0.2f,
            float gamma = 0.9f) : base(env, "SARSA")
        {
            Epsilon = epsilon;
            Alpha = alpha;
            Gamma = gamma;
        }

        public float Alpha { protected set; get; }
        public float Gamma { protected set; get; }

        public Episode Learn()
        {
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            var act = GetEpsilonAct(Environ.Observation!.Value!);
            while (Environ.IsComplete(epoch) == false)
            {
                epoch++;
                var step = Environ.Step(act, epoch);
                act = Update(step);
                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);
                Environ.Observation = step.StateNew; /// It's import for Update Observation
            }

            var orginalReward = episode.Steps.Sum(a => a.Reward.Value);
            var sumReward = orginalReward;
            episode.SumReward = new Reward(sumReward);
            return episode;
        }


        public Act Update(Step step)
        {
            var state = step.State.Value!;
            var action = step.Action.Value!;
            var stateNew = step.StateNew.Value!;
            var reward = step.Reward.Value;

            var currentTransit = new TransitKey(state, action);

            var bestValue = ValueTable.GetBestValue(stateNew);
            var nextAct = GetEpsilonAct(stateNew);
            var newValue = reward + Gamma * bestValue;
            var oldValue = ValueTable[currentTransit];
            var finalValue = oldValue * (1 - Alpha) + newValue * Alpha;

            ValueTable[currentTransit] = finalValue;
            return nextAct;
        }
    }
}
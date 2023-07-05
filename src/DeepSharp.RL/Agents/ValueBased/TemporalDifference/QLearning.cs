using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class QLearning : ValueAgent
    {
        /// <summary>
        /// </summary>
        /// <param name="env"></param>
        /// <param name="epsilon">epsilon of ε-greedy Policy</param>
        /// <param name="alpha">learning rate</param>
        /// <param name="gamma">rate of discount</param>
        public QLearning(Environ<Space, Space> env,
            float epsilon = 0.1f,
            float alpha = 0.2f,
            float gamma = 0.9f) :
            base(env, "QLearning")
        {
            Epsilon = epsilon;
            Alpha = alpha;
            Gamma = gamma;
        }


        public float Alpha { protected set; get; }
        public float Gamma { protected set; get; }


        public override Act GetPolicyAct(torch.Tensor state)
        {
            var action = QTable.GetBestAct(state);
            return action ?? GetSampleAct();
        }


        public override LearnOutcome Learn()
        {
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            while (Environ.IsComplete(epoch) == false)
            {
                epoch++;
                var epsilonAct = GetEpsilonAct(Environ.Observation!.Value!);
                var step = Environ.Step(epsilonAct, epoch);

                Update(step);

                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);

                Environ.Observation = step.PostState; /// It's import for Update Observation
            }

            var sumReward = episode.Steps.Sum(a => a.Reward.Value);
            episode.SumReward = new Reward(sumReward);

            return new LearnOutcome(episode);
        }

        public void Update(Step step)
        {
            var s = step.PreState.Value!;
            var a = step.Action.Value!;
            var r = step.Reward.Value;
            var sNext = step.PostState.Value!;
            var q = QTable[s, a];

            var aNext = GetPolicyAct(sNext);
            var qNext = QTable[sNext, aNext.Value!];

            QTable[s, a] = q + Alpha * (r + Gamma * qNext - q);
        }
    }
}
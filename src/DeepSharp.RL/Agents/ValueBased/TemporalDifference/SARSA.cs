using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class SARSA : ValueAgent
    {
        /// <summary>
        /// </summary>
        /// <param name="env"></param>
        /// <param name="epsilon">epsilon of ε-greedy Policy</param>
        /// <param name="alpha">learning rate</param>
        /// <param name="gamma">rate of discount</param>
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

        public override LearnOutcome Learn()
        {
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            var act = GetEpsilonAct(Environ.Observation!.Value!);
            while (Environ.IsComplete(epoch) == false)
            {
                epoch++;
                var step = Environ.Step(act, epoch);

                var actNext = Update(step); ///

                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);

                Environ.Observation = step.PostState; /// It's import for Update Observation
                act = actNext;
            }

            var sumReward = episode.Steps.Sum(a => a.Reward.Value);
            episode.SumReward = new Reward(sumReward);

            return new LearnOutcome(episode);
        }


        public Act Update(Step step)
        {
            var s = step.PreState.Value!;
            var a = step.Action.Value!;
            var r = step.Reward.Value;
            var sNext = step.PostState.Value!;
            var q = QTable[s, a];

            var aNext = GetEpsilonAct(sNext); /// a' by ε-greedy policy
            var qNext = QTable[sNext, aNext.Value!];


            QTable[s, a] = q + Alpha * (r + Gamma * qNext - q);
            return aNext;
        }
    }
}
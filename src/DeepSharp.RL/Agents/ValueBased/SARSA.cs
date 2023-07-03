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

                var actNext = Update(step); ///

                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);

                Environ.Observation = step.StateNew; /// It's import for Update Observation
                act = actNext;
            }

            var orginalReward = episode.Steps.Sum(a => a.Reward.Value);
            var sumReward = orginalReward;
            episode.SumReward = new Reward(sumReward);
            return episode;
        }


        public Act Update(Step step)
        {
            var s = step.State.Value!;
            var a = step.Action.Value!;
            var r = step.Reward.Value;
            var sNext = step.StateNew.Value!;
            var q = ValueTable[s, a];

            var aNext = GetEpsilonAct(sNext); /// a' by ε-greedy policy
            var qNext = ValueTable[sNext, aNext.Value!];


            ValueTable[s, a] = q + Alpha * (r + Gamma * qNext - q);
            return aNext;
        }
    }
}
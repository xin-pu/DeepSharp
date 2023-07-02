using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     Monte Carlo Method On Policy
    /// </summary>
    public class MonteCarloOnPolicy : ValueAgent
    {
        /// <summary>
        /// </summary>
        /// <param name="env"></param>
        /// <param name="t">the leaning count of  each epoch </param>
        public MonteCarloOnPolicy(Environ<Space, Space> env, float epsilon = 0.8f, int t = 10)
            : base(env, "MonteCarloOnPolicy")
        {
            Epsilon = epsilon;
            T = t;
            Count = new Dictionary<TransitKey, int>();
        }

        public int T { protected set; get; }

        public Dictionary<TransitKey, int> Count { set; get; }


        public Episode Learn()
        {
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            var act = GetEpsilonAct(Environ.Observation!.Value!);
            while (Environ.IsComplete(epoch) == false && epoch < T)
            {
                epoch++;
                var step = Environ.Step(act, epoch);

                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);
                Environ.Observation = step.StateNew; /// It's import for Update Observation
            }

            Update(episode);

            var orginalReward = episode.Steps.Sum(a => a.Reward.Value);
            var sumReward = orginalReward;
            episode.SumReward = new Reward(sumReward);
            return episode;
        }


        private int GetCount(TransitKey transitKey)
        {
            if (Count.ContainsKey(transitKey) == false)
                Count[transitKey] = 0;

            return Count[transitKey];
        }

        public void Update(Episode episode)
        {
            var lenth = episode.Length;
            var steps = episode.Steps;
            foreach (var t in Enumerable.Range(0, lenth))
            {
                var step = steps[t];
                var key = new TransitKey(step.State, step.Action);
                var r = steps.Skip(t).Average(a => a.Reward.Value);
                ValueTable[key] = (ValueTable[key] * GetCount(key) + r) / (GetCount(key) + 1);
                Count[key] = GetCount(key) + 1;
            }
        }
    }
}
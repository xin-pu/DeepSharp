using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class MonteCarloOffPolicy : ValueAgent
    {
        public MonteCarloOffPolicy(Environ<Space, Space> env, float epsilon, int t)
            : base(env, "MonteCarloOffPolicy")
        {
            Epsilon = epsilon;
            T = t;
            Count = new Dictionary<TransitKey, int>();
        }

        public int T { protected set; get; }

        public Dictionary<TransitKey, int> Count { protected set; get; }


        public Episode Learn()
        {
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            var act = GetPolicyAct(Environ.Observation!.Value!);
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


        public void Update(Episode episode)
        {
            var lenth = episode.Length;
            var steps = episode.Steps;
            foreach (var t in Enumerable.Range(0, lenth))
            {
                var step = steps[t];
                var key = new TransitKey(step.State, step.Action);
                var r = steps.Skip(t).Average(a => a.Reward.Value);
                var per = steps.Skip(t).Select(GetTransitPer).Aggregate(1f, (a, b) => a * b);
                var finalR = r * per;
                ValueTable[key] = (ValueTable[key] * GetCount(key) + finalR) / (GetCount(key) + 1);
                Count[key] = GetCount(key) + 1;
            }
        }


        private float GetTransitPer(Step step)
        {
            var actPolicy = GetPolicyAct(step.State.Value!).Value!;
            var actStep = step.Action.Value!;
            var actionSpace = Environ.ActionSpace!.N;
            var e = actPolicy.Equals(actStep) ? 1 : 0;
            var per = actPolicy.Equals(actStep)
                ? 1 - Epsilon + Epsilon / actionSpace
                : Epsilon / actionSpace;
            return per / e;
        }

        private int GetCount(TransitKey transitKey)
        {
            if (Count.ContainsKey(transitKey) == false)
                Count[transitKey] = 0;

            return Count[transitKey];
        }
    }
}
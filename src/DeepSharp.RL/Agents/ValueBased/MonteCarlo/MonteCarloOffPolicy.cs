using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     Todo has issue about GetTransitPer?   2023/7/3
    /// </summary>
    public class MonteCarloOffPolicy : ValueAgent
    {
        public MonteCarloOffPolicy(Environ<Space, Space> env, float epsilon = 0.1f, int t = 10)
            : base(env, "MonteCarloOffPolicy")
        {
            Epsilon = epsilon;
            T = t;
            Count = new Dictionary<TransitKey, int>();
        }

        public int T { protected set; get; }

        public Dictionary<TransitKey, int> Count { protected set; get; }


        public override LearnOutcome Learn()
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
                Environ.Observation = step.PostState; /// It's import for Update Observation
            }

            Update(episode);

            var sumReward = episode.Steps.Sum(a => a.Reward.Value);
            episode.SumReward = new Reward(sumReward);

            var learnOut = new LearnOutcome(episode);

            return learnOut;
        }


        public void Update(Episode episode)
        {
            var lenth = episode.Length;
            var steps = episode.Steps;
            foreach (var t in Enumerable.Range(0, lenth))
            {
                var step = steps[t];
                var key = new TransitKey(step.PreState, step.Action);
                var r = steps.Skip(t).Average(a => a.Reward.Value);
                var per = steps.Skip(t).Select(GetTransitPer).Aggregate(1f, (a, b) => a * b); ///Error Here
                var finalR = r * per;
                var count = GetCount(key);
                QTable[key] = (QTable[key] * count + finalR) / (count + 1);
                SetCount(key, count + 1);
            }
        }


        private float GetTransitPer(Step step)
        {
            var actPolicy = GetPolicyAct(step.PreState.Value!).Value!;
            var actStep = step.Action.Value!;
            var actionSpace = Environ.ActionSpace!.N;
            var e = 1f;
            var per = actPolicy.Equals(actStep)
                ? 1 - Epsilon + Epsilon / actionSpace
                : Epsilon / actionSpace;
            return e / per;
        }

        private int GetCount(TransitKey transitKey)
        {
            Count.TryAdd(transitKey, 0);
            return Count[transitKey];
        }

        private void SetCount(TransitKey transitKey, int value)
        {
            Count[transitKey] = value;
        }
    }
}
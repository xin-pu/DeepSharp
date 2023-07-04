using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class ValueIterate : ValueAgent
    {
        /// <summary>
        /// </summary>
        /// <param name="env"></param>
        /// <param name="transitionProbs">转移概率表</param>
        /// <param name="transitionRewards">转移奖励表</param>
        /// <param name="t">T</param>
        public ValueIterate(Environ<Space, Space> env,
            Dictionary<RewardKey, float> transitionProbs,
            Dictionary<RewardKey, float> transitionRewards,
            int t = 100)
            : base(env, "ValueIterate")
        {
            T = t;
            TransitionProbs = transitionProbs;
            TransitionRewards = transitionRewards;
            RewardKeys = transitionProbs.Keys.ToArray();
            X = TransitionProbs.Keys.Select(a => a.State!)
                .Distinct()
                .ToArray();

            V = X.ToDictionary(p => p, p => 0f);
        }

        public int T { protected set; get; }

        public Dictionary<RewardKey, float> TransitionProbs { protected set; get; }
        public Dictionary<RewardKey, float> TransitionRewards { protected set; get; }

        public Dictionary<torch.Tensor, float> V { protected set; get; }
        public torch.Tensor[] X { protected set; get; }
        public RewardKey[] RewardKeys { protected set; get; }


        public override Episode Learn()
        {
            var vNext = GetVTable();
            QTable = GetQTable(TransitionProbs, TransitionRewards, vNext, T);
            return new Episode();
        }

        protected Dictionary<torch.Tensor, float> GetVTable()
        {
            var vNext = X.ToDictionary(p => p, p => 0f);

            foreach (var t in Enumerable.Range(0, T))
            foreach (var x in X)
                vNext[x] = RewardKeys
                    .Where(a => a.State.Equals(x))
                    .Sum(r => TransitionProbs[r] * (TransitionRewards[r] / t + vNext[r.NewState] * (t - 1) / t));
            return vNext;
        }

        protected QTable GetQTable(
            Dictionary<RewardKey, float> p,
            Dictionary<RewardKey, float> r,
            Dictionary<torch.Tensor, float> v,
            int t)
        {
            var q = new QTable();

            var x = TransitionProbs.Keys
                .Select(a => a.State!)
                .Distinct()
                .ToArray();

            var rewardKeys = p.Keys
                .ToArray();


            return q;
        }
    }
}
using DeepSharp.RL.Environs;
using DeepSharp.Utility;

namespace DeepSharp.RL.Agents
{
    public abstract class PolicyIteration : ValueAgent
    {
        /// <summary>
        /// </summary>
        /// <param name="env"></param>
        /// <param name="p"></param>
        /// <param name="r"></param>
        /// <param name="t"></param>
        /// <param name="threshold"></param>
        protected PolicyIteration(Environ<Space, Space> env, Dictionary<RewardKey, float> p,
            Dictionary<RewardKey, float> r, int t = 100)
            : base(env, "ValueIterate")
        {
            T = t;
            VTable = new VTable();
            P = p;
            R = r;
            RewardKeys = p.Keys.ToArray();
            X = P.Keys.Select(a => a.State)
                .Distinct(new TensorEqualityCompare())
                .ToArray();
        }

        public int T { protected set; get; }


        public VTable VTable { protected set; get; }

        protected Dictionary<RewardKey, float> P { set; get; }
        protected Dictionary<RewardKey, float> R { set; get; }
        protected torch.Tensor[] X { set; get; }
        protected RewardKey[] RewardKeys { set; get; }


        public override LearnOutcome Learn()
        {
            while (true)
            {
                /// Update VTable
                foreach (var t in Enumerable.Range(1, T))
                {
                    var vNext = GetVTable(t);
                    VTable = vNext;
                }

                /// Policy Iterate
                /// Get Policy (argmax Q=> Update QTable) by Value
                var qTable = GetQTable(VTable, T);
                if (qTable == QTable) break;
                QTable = qTable;
            }

            return new LearnOutcome();
        }

        protected abstract VTable GetVTable(int t);

        protected abstract QTable GetQTable(VTable vTable, int t);
    }
}
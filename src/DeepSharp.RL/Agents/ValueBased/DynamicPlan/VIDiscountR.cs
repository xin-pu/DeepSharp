using DeepSharp.RL.Environs;
using DeepSharp.Utility;

namespace DeepSharp.RL.Agents
{
    public class VIDiscountR : ValueIterate
    {
        public VIDiscountR(Environ<Space, Space> env, Dictionary<RewardKey, float> p,
            Dictionary<RewardKey, float> r, int t, float gamma = 0.9f, float threshold = 0.1f)
            : base(env, p, r, t, threshold)
        {
            Gamma = gamma;
        }

        public float Gamma { protected set; get; }

        protected override VTable GetVTable(int t)
        {
            var vNext = new VTable();

            foreach (var unused in Enumerable.Range(0, t))
            foreach (var x in X)
                vNext[x] = RewardKeys
                    .Where(a => a.State.Equals(x))
                    .Sum(r => P[r] * (R[r] + vNext[r.NewState] * Gamma));
            return vNext;
        }

        protected override QTable GetQTable(VTable v, int t)
        {
            var q = new QTable();

            var states = P.Keys
                .Select(a => a.State)
                .Distinct(new TensorEqualityCompare())
                .ToArray();

            var actions = P.Keys
                .Select(a => a.Act)
                .Distinct(new TensorEqualityCompare())
                .ToArray();

            foreach (var state in states)
            foreach (var action in actions)
            {
                var value = RewardKeys.Where(a => a.State.Equals(state) && a.Act.Equals(action))
                    .Sum(a => P[a] * (R[a] + v[a.NewState] * Gamma));
                q[state, action] = value;
            }

            return q;
        }
    }
}
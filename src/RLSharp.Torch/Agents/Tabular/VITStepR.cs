using RLSharp.Torch.Environs;
using RLSharp.Torch;

namespace RLSharp.Torch.Agents.Tabular
{
	public class VITStepR : ValueIterate
	{
		public VITStepR(EnvironmentBase<Space, Space> env,
			Dictionary<RewardKey, float>      p,
			Dictionary<RewardKey, float>      r,
			int                               t         = 100,
			float                             threshold = 0.1f) : base(env, p, r, t, threshold)
		{
		}

		protected override VTable GetVTable(int t)
		{
			var vNext = new VTable();

			foreach (var i in Enumerable.Range(1, t))
			foreach (var x in X)
				vNext[x] = RewardKeys
					.Where(a => a.State.Equals(x))
					.Sum(r => P[r] * (R[r] / i + vNext[r.NewState] * (i - 1) / i));
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
				.Select(a => a.ActionValue)
				.Distinct(new TensorEqualityCompare())
				.ToArray();

			foreach (var state in states)
			foreach (var action in actions)
			{
				var value = RewardKeys.Where(a => a.State.Equals(state) && a.ActionValue.Equals(action))
					.Sum(a => P[a] * (R[a] / t + v[a.NewState] * (t - 1) / t));
				q[state, action] = value;
			}

			return q;
		}
	}
}
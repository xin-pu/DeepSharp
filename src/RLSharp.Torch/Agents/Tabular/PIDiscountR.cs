using RLSharp.Torch.Environs;
using RLSharp.Torch;

namespace RLSharp.Torch.Agents.Tabular
{
	public class PIDiscountR : PolicyIteration
	{
		public PIDiscountR(EnvironmentBase<Space, Space> env,
			Dictionary<RewardKey, float>         p,
			Dictionary<RewardKey, float>         r,
			int                                  t,
			float                                gamma = 0.9f)
			: base(env, p, r, t)
		{
			Gamma = gamma;
		}

		public float Gamma { get; protected set; }

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
				.Select(a => a.ActionValue)
				.Distinct(new TensorEqualityCompare())
				.ToArray();

			foreach (var state in states)
			foreach (var action in actions)
			{
				var value = RewardKeys.Where(a => a.State.Equals(state) && a.ActionValue.Equals(action))
					.Sum(a => P[a] * (R[a] + v[a.NewState] * Gamma));
				q[state, action] = value;
			}

			return q;
		}
	}
}
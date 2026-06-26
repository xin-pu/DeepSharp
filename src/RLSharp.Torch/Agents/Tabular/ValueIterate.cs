using RLSharp.Torch.Environs;
using RLSharp.Torch;

namespace RLSharp.Torch.Agents.Tabular
{
	public abstract class ValueIterate : TabularAgent
	{
		/// <summary>
		/// </summary>
		/// <param name="env"></param>
		/// <param name="p"></param>
		/// <param name="r"></param>
		/// <param name="t"></param>
		/// <param name="threshold"></param>
		protected ValueIterate(EnvironmentBase<Space, Space> env,
			Dictionary<RewardKey, float>             p,
			Dictionary<RewardKey, float>             r,
			int                                      t         = 100,
			float                                    threshold = 0.1f)
			: base(env, "ValueIterate")
		{
			T          = t;
			Threshold  = threshold;
			VTable     = new VTable();
			P          = p;
			R          = r;
			RewardKeys = p.Keys.ToArray();
			X = P.Keys.Select(a => a.State)
				.Distinct(new TensorEqualityCompare())
				.ToArray();
		}

		public int T { get; protected set; }

		/// <summary>
		///     Convergence threshold.
		/// </summary>
		public float Threshold { get; protected set; }

		public VTable VTable { get; protected set; }

		protected Dictionary<RewardKey, float> P { get; set; }

		protected Dictionary<RewardKey, float> R { get; set; }

		protected torch.Tensor[] X { get; set; }

		protected RewardKey[] RewardKeys { get; set; }


		public override LearnOutcome Learn()
		{
			// Value Iteration loop
			foreach (var t in Enumerable.Range(1, T))
			{
				var vNext = GetVTable(t);
				if (vNext - VTable < Threshold)
					break;
				VTable = vNext;
			}

			// Get Policy (argmax Q => Update QTable) from Value
			var qTable = GetQTable(VTable, T);
			QTable = qTable;
			return new LearnOutcome();
		}

		protected abstract VTable GetVTable(int t);

		protected abstract QTable GetQTable(VTable vTable, int t);
	}
}
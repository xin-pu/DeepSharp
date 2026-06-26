using RLSharp.Torch.Environs;
using RLSharp.Torch.ExpReplays;

namespace RLSharp.Torch.Agents.Deep.Value
{
	/// <summary>
	///     Categorical DQN (C51)
	///     Learns the probability distribution of action values rather than scalar Q values.
	///     Uses numAtoms=51 discrete support points covering [V_min, V_max],
	///     minimizing the cross-entropy between predicted distribution and projected Bellman target distribution.
	/// </summary>
	public class CategoricalDQN : DeepValueAgent
	{
		public CategoricalDQN(EnvironmentBase<Space, Space> env,
			int                                     n         = 1000,
			int                                     c         = 10000,
			float                                   gamma     = 0.99f,
			int                                     batchSize = 32,
			long                                    numAtoms  = 51,
			float                                   vMin      = -10f,
			float                                   vMax      = 10f)
			: base(env, "CategoricalDQN")
		{
			C         = c;
			N         = n;
			BatchSize = batchSize;
			Gamma     = gamma;
			NumAtoms  = numAtoms;
			VMin      = vMin;
			VMax      = vMax;
			DeltaZ    = (VMax - VMin) / (numAtoms - 1);
			Support   = torch.linspace(VMin, VMax, numAtoms);
			Epsilon   = 0.1f;

			Q       = new CategoricalNet(ObservationSize, 128, ActionSize, NumAtoms);
			QTarget = new CategoricalNet(ObservationSize, 128, ActionSize, NumAtoms);
			QTarget.load_state_dict(Q.state_dict());

			Optimizer  = SGD(Q.parameters(), 0.001);
			Loss       = MSELoss();
			UniformExp = new UniformExpReplay(C);
		}

		/// <summary>
		///     Number of value distribution support points (51 for C51).
		/// </summary>
		public long NumAtoms { get; }

		/// <summary>
		///     Lower bound of the value range.
		/// </summary>
		public float VMin { get; }

		/// <summary>
		///     Upper bound of the value range.
		/// </summary>
		public float VMax { get; }

		/// <summary>
		///     Support point spacing: Δz = (VMax - VMin) / (NumAtoms - 1).
		/// </summary>
		public float DeltaZ { get; }

		/// <summary>
		///     Support points: z_i = VMin + i * Δz.
		/// </summary>
		public torch.Tensor Support { get; }

		public float Gamma { get; }

		public int C { get; }

		public int N { get; }

		public int BatchSize { get; }

		public Module<torch.Tensor, torch.Tensor> QTarget { get; protected set; }

		public UniformExpReplay UniformExp { get; }

		/// <summary>
		///     Get Q values from logits: Q(s,a) = Σ p_i(s,a) * z_i.
		/// </summary>
		private torch.Tensor GetQValues(torch.Tensor logits)
		{
			// logits: [batch, actionSize, numAtoms]
			var probs = functional.softmax(logits, -1);      // [batch, actionSize, numAtoms]
			return (probs * Support).sum(new long[] { -1 }); // [batch, actionSize]
		}

		/// <summary>
		///     C51 learning loop.
		/// </summary>
		public override LearnOutcome Learn()
		{
			var learnOutCome = new LearnOutcome();
			foreach (var _ in Enumerable.Range(0, N))
			{
				EnvironmentBase.Reset();
				var epoch   = 0;
				var episode = new Episode();
				while (!EnvironmentBase.IsComplete(epoch))
				{
					epoch++;
					var ActionValue  = GetEpsilonAct(EnvironmentBase.ObservationValue!.Value!);
					var step = EnvironmentBase.Step(ActionValue, epoch);
					episode.Enqueue(step);

					EnvironmentBase.CallBack?.Invoke(step);
					EnvironmentBase.ObservationValue = step.PostState;
				}

				learnOutCome.AppendStep(episode);
				UniformExp.Enqueue(episode);
				if (UniformExp.Buffers.Count >= C)
					learnOutCome.Evaluate = UpdateNet();
			}

			SyncTargetNetwork();
			return learnOutCome;
		}

		/// <summary>
		///     C51 distributional Bellman update.
		///     Projects the Bellman target onto discrete support points, minimizing cross-entropy with the predicted distribution.
		/// </summary>
		private float UpdateNet()
		{
			var batchSample = UniformExp.Sample(BatchSize);
			var batch       = batchSample.PreState.shape[0];

			// 1. Current Q distribution
			var predLogits = Q.forward(batchSample.PreState); // [batch, actionSize, numAtoms]
			var predProbs  = functional.softmax(predLogits, -1);

			// 2. Target Q distribution (computed using QTarget)
			using var _            = torch.no_grad();
			var       targetLogits = QTarget.forward(batchSample.PostState); // [batch, actionSize, numAtoms]
			var       targetProbs  = functional.softmax(targetLogits, -1);
			var       targetQ      = (targetProbs * Support).sum(new long[] { -1 }); // [batch, actionSize]

			// a* = argmax_a QTarget(s', a)
			var bestActions = targetQ.argmax(1); // [batch]

			// Probability distribution of the optimal action p(s', a*)
			var bestProbs = targetProbs[torch.arange(batch), bestActions]; // [batch, numAtoms]

			// 3. Project Bellman update onto support points
			// T_z_j = r + γ * z_j, clipped to [VMin, VMax]
			var tz = batchSample.Reward.unsqueeze(-1) + Gamma * Support.unsqueeze(0); // [batch, numAtoms]
			tz = tz.clamp(VMin, VMax);

			// b_j = (T_z_j - VMin) / Δz
			var b = (tz - VMin) / DeltaZ; // [batch, numAtoms]

			var l = b.floor().to_type(torch.ScalarType.Int64); // [batch, numAtoms]
			var u = b.ceil().to_type(torch.ScalarType.Int64);  // [batch, numAtoms]

			// Project target distribution onto discrete support points (manual loop, batch usually �?56)
			var targetDistribution = torch.zeros(batch, NumAtoms);
			var massL              = bestProbs * (u.to_type(torch.ScalarType.Float32) - b); // [batch, numAtoms]
			var massU              = bestProbs * (b - l.to_type(torch.ScalarType.Float32)); // [batch, numAtoms]

			for (var i = 0; i < batch; i++)
			for (var j = 0; j < NumAtoms; j++)
			{
				var li                                                  = l[i, j].ToInt64();
				var ui                                                  = u[i, j].ToInt64();
				if (li >= 0 && li < NumAtoms) targetDistribution[i, li] += massL[i, j];
				if (ui >= 0 && ui < NumAtoms) targetDistribution[i, ui] += massU[i, j];
			}

			// 4. Cross-entropy loss: -Σ targetDistribution * log(predProbs[action])
			var actionIndices = batchSample.Action.squeeze(-1).to_type(torch.ScalarType.Int64); // [batch]
			var predictedDist = predProbs[torch.arange(batch), actionIndices];                  // [batch, numAtoms]

			var loss = -(targetDistribution * (predictedDist + 1e-8f).log()).sum(1).mean();

			Optimizer.zero_grad();
			loss.backward();
			Optimizer.step();
			return loss.item<float>();
		}

		private void SyncTargetNetwork()
		{
			var parameters = Q.state_dict();
			QTarget.load_state_dict(parameters);
		}
	}
}
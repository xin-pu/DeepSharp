using RLSharp.Torch.Environs;
using RLSharp.Torch.ExpReplays;

namespace RLSharp.Torch.Agents.Deep.Policy
{
	/// <summary>
	///     Proximal Policy Optimization for discrete action spaces.
	/// </summary>
	public class PPO : DeepPolicyAgent
	{
		public PPO(EnvironmentBase<Space, Space> env,
			int    batchSize    = 4,
			float  gamma        = 0.99f,
			float  learningRate = 0.001f,
			float  clipEpsilon  = 0.2f,
			int    updateEpochs = 4)
			: base(env, "PPO")
		{
			if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
			if (updateEpochs <= 0) throw new ArgumentOutOfRangeException(nameof(updateEpochs));
			if (clipEpsilon <= 0) throw new ArgumentOutOfRangeException(nameof(clipEpsilon));

			BatchSize    = batchSize;
			Gamma        = gamma;
			LearningRate = learningRate;
			ClipEpsilon  = clipEpsilon;
			UpdateEpochs = updateEpochs;

			ExpReplays = new EpisodeExpReplay(batchSize, gamma);
			Optimizer  = Adam(PolicyNet.parameters(), LearningRate);
		}

		public int BatchSize { get; }

		public float Gamma { get; }

		public float LearningRate { get; }

		public float ClipEpsilon { get; }

		public int UpdateEpochs { get; }

		public EpisodeExpReplay ExpReplays { get; }

		public override LearnOutcome Learn()
		{
			var outcome = new LearnOutcome();
			var episodes = RunEpisodes(BatchSize);

			foreach (var episode in episodes)
			{
				outcome.AppendStep(episode);
				ExpReplays.Enqueue(episode);
			}

			using var batch = ExpReplays.All();
			ExpReplays.Clear();

			using var oldLogProb = torch.log(PolicyNet.forward(batch.PreState))
				.gather(1, batch.Action)
				.detach();

			torch.Tensor? lastLoss = null;
			for (var epoch = 0; epoch < UpdateEpochs; epoch++)
			{
				Optimizer.zero_grad();

				var logProb = torch.log(PolicyNet.forward(batch.PreState)).gather(1, batch.Action);
				var ratio = torch.exp(logProb - oldLogProb);
				var clippedRatio = torch.clamp(ratio, 1f - ClipEpsilon, 1f + ClipEpsilon);
				var objective = torch.min(ratio * batch.Reward, clippedRatio * batch.Reward);
				var loss = -objective.mean();

				loss.backward();
				Optimizer.step();

				lastLoss?.Dispose();
				lastLoss = loss.detach();
			}

			outcome.Evaluate = lastLoss?.item<float>() ?? 0f;
			lastLoss?.Dispose();
			return outcome;
		}
	}
}

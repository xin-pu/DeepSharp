using RLSharp.Core.Agents;

namespace RLSharp.Core.Training
{
	public sealed class Trainer<TState, TAction>
	{
		public Trainer(IAgent<TState, TAction> agent)
		{
			Agent = agent;
		}

		public IAgent<TState, TAction> Agent { get; }

		public event EventHandler<int>? EpisodeStarted;

		public event EventHandler<LearnResult>? EpisodeCompleted;

		public TrainingResult Train(TrainingOptions? options = null, CancellationToken cancellationToken = default)
		{
			options ??= new TrainingOptions();
			if (options.MaxEpisodes <= 0)
				throw new ArgumentOutOfRangeException(nameof(options.MaxEpisodes));

			var results = new List<LearnResult>(options.MaxEpisodes);
			var steps   = 0;

			for (var episode = 1; episode <= options.MaxEpisodes; episode++)
			{
				cancellationToken.ThrowIfCancellationRequested();
				EpisodeStarted?.Invoke(this, episode);

				var result = Agent.Learn();
				steps += result.Steps;
				results.Add(result);
				EpisodeCompleted?.Invoke(this, result);

				if (ShouldStop(results, options, episode))
					break;
			}

			string? savedPath = null;
			if (!string.IsNullOrWhiteSpace(options.CheckpointPath))
			{
				Agent.Save(options.CheckpointPath);
				savedPath = options.CheckpointPath;
			}

			return new TrainingResult(
				results.Count,
				steps,
				results.Count == 0 ? 0 : results.Average(r => r.Reward),
				savedPath);
		}

		private static bool ShouldStop(IReadOnlyList<LearnResult> results, TrainingOptions options, int episode)
		{
			if (options.StopAverageReward is null)
				return false;
			if (options.ValidationEpisodes <= 0 || episode % options.ValidationInterval != 0)
				return false;

			var window = results.TakeLast(options.ValidationEpisodes).ToArray();
			return window.Length > 0 && window.Average(r => r.Reward) >= options.StopAverageReward.Value;
		}
	}
}
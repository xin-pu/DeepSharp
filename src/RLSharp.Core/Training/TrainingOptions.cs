namespace RLSharp.Core.Training
{
	public sealed record TrainingOptions
	{
		public int MaxEpisodes { get; init; } = 1_000;

		public int ValidationEpisodes { get; init; } = 10;

		public int ValidationInterval { get; init; } = 10;

		public float? StopAverageReward { get; init; }

		public string? CheckpointPath { get; init; }
	}
}
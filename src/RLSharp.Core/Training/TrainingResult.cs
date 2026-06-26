namespace RLSharp.Core.Training
{
	public sealed record TrainingResult(
		int     Episodes,
		int     Steps,
		float   AverageReward,
		string? SavedPath = null);
}
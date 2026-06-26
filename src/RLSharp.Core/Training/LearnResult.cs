namespace RLSharp.Core.Training
{
	public sealed record LearnResult(
		int    Steps,
		float  Reward,
		float? Loss = null);
}
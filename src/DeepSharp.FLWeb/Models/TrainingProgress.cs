namespace DeepSharp.FLWeb.Models
{
	public class TrainingProgress
	{
		public int EpisodeCount { get; set; }

		public int StepCount { get; set; }

		public float SumReward { get; set; }

		public float AverageReward { get; set; }

		public float Epsilon { get; set; }

		public float Loss { get; set; }
	}
}
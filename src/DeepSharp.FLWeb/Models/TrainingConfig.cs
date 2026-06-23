namespace DeepSharp.FLWeb.Models
{
	public class TrainingConfig
	{
		private static readonly HashSet<string> SupportedAgents =
		[
			"QLearning", "SARSA", "MonteCarloOnPolicy", "MonteCarloOffPolicy", "DQN", "REINFORCE", "A2C"
		];

		// Agent selection
		public string AgentType { get; set; } = "QLearning";

		// Common hyperparameters
		public float Epsilon { get; set; } = 0.2f;

		public float Gamma { get; set; } = 0.9f;

		public float Alpha { get; set; } = 0.2f;

		// Monte Carlo
		public int T { get; set; } = 50;

		// DQN
		public int N { get; set; } = 1000;

		public int C { get; set; } = 10000;

		public int BatchSize { get; set; } = 32;

		// A2C
		public float Beta { get; set; } = 0.01f;

		// FrozenLake smooth probabilities
		public float SmoothTarget { get; set; } = 0.8f;

		public float SmoothLeft { get; set; } = 0.1f;

		public float SmoothRight { get; set; } = 0.1f;

		// Training control
		public int MaxEpisodes { get; set; } = 1000;

		public int SpeedDelayMs { get; set; } = 50;

		public void Validate()
		{
			if (!SupportedAgents.Contains(AgentType))
				throw new ArgumentException($"Unknown agent type: {AgentType}", nameof(AgentType));
			ValidateUnitInterval(Epsilon, nameof(Epsilon));
			ValidateUnitInterval(Gamma, nameof(Gamma));
			ValidateUnitInterval(Alpha, nameof(Alpha));
			ValidateUnitInterval(Beta, nameof(Beta));
			ValidateUnitInterval(SmoothTarget, nameof(SmoothTarget));
			ValidateUnitInterval(SmoothLeft, nameof(SmoothLeft));
			ValidateUnitInterval(SmoothRight, nameof(SmoothRight));

			var smoothingTotal = SmoothTarget + SmoothLeft + SmoothRight;
			if (Math.Abs(smoothingTotal - 1f) > 0.0001f)
				throw new ArgumentException("FrozenLake smoothing probabilities must sum to 1.");
			if (T <= 0 || N <= 0 || C <= 0 || BatchSize <= 0)
				throw new ArgumentException("T, N, C, and BatchSize must be positive.");
			if (BatchSize > C)
				throw new ArgumentException("BatchSize cannot exceed replay capacity C.");
			if (MaxEpisodes is <= 0 or > 1_000_000)
				throw new ArgumentOutOfRangeException(nameof(MaxEpisodes),
					"MaxEpisodes must be between 1 and 1,000,000.");
			if (SpeedDelayMs is < 0 or > 60_000)
				throw new ArgumentOutOfRangeException(nameof(SpeedDelayMs),
					"SpeedDelayMs must be between 0 and 60,000.");
		}

		private static void ValidateUnitInterval(float value, string name)
		{
			if (!float.IsFinite(value) || value is < 0f or > 1f)
				throw new ArgumentOutOfRangeException(name, "Value must be finite and between 0 and 1.");
		}
	}
}
namespace DeepSharp.FLWeb.Models;

public class TrainingConfig
{
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
}

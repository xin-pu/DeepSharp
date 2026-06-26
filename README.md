# RLSharp

RLSharp is a reinforcement learning project built with .NET 8 and TorchSharp.
It is currently intended for learning, experimentation, and further development.
The public API is still allowed to change.

## Features

- Tabular reinforcement learning algorithms.
- Deep reinforcement learning algorithms backed by TorchSharp.
- FrozenLake and K-Armed Bandit environments.
- Action selectors, replay buffers, trainers, and network helpers.
- FrozenLake web visualizer.
- Console sample and xUnit test suite.

## Requirements

- .NET 8 SDK
- Windows, Linux, or macOS
- CPU TorchSharp runtime through `TorchSharp-cpu`

## Project Structure

```text
RLSharp
|-- src
|   |-- RLSharp.Core                 Shared training abstractions
|   |-- RLSharp.Torch                TorchSharp-backed RL library
|   |-- RLSharp.FrozenLake.Web       FrozenLake ASP.NET Core visualizer
|   |-- RLSharp.Samples.Console      Console sample
|   |-- RLSharp.Tests                xUnit tests
|   `-- RLSharp.sln
|-- images
`-- resources
```

`RLSharp.Torch` currently contains the environment and agent implementations.
`RLSharp.Core` is kept TorchSharp-free for shared abstractions; future work can
move pure .NET tabular/environment contracts there as the API stabilizes.

## Build

```powershell
dotnet restore src/RLSharp.sln
dotnet build src/RLSharp.sln --no-restore
```

## Quick Start

```csharp
using RLSharp.Torch;
using RLSharp.Torch.Agents.Tabular;
using RLSharp.Torch.Environs;

RandomProvider.SetSeed(42);

var environment = new FrozenLake([0.8f, 0.1f, 0.1f]);
var agent = new QLearning(environment, epsilon: 0.2f, alpha: 0.2f, gamma: 0.9f);

for (var episode = 0; episode < 1_000; episode++)
{
    agent.Learn();
}

var result = agent.RunEpisode();
Console.WriteLine($"Reward: {result.SumReward.Value}");
```

## Included Algorithms

| Category | Implementations |
| --- | --- |
| Tabular methods | Q-Learning, SARSA, on-policy Monte Carlo, off-policy Monte Carlo |
| Dynamic programming | Policy Iteration, Value Iteration, and variants |
| Value-based deep RL | DQN, Double DQN, Dueling DQN, Noisy DQN, Categorical DQN, CGP |
| Policy gradient | REINFORCE, Cross Entropy, PPO |
| Actor-Critic | Actor-Critic, A2C, A3C |
| Continuous control boundaries | DDPG, TD3, SAC |

DDPG, TD3, and SAC are present as Torch-layer algorithm boundaries for the next
continuous-control phase. They are not exposed in the FrozenLake web UI because
FrozenLake uses a discrete action space.

The FrozenLake web UI currently supports:

- `QLearning`
- `SARSA`
- `MonteCarloOnPolicy`
- `MonteCarloOffPolicy`
- `DQN`
- `REINFORCE`
- `A2C`
- `PPO`

## Running Samples

Console sample:

```powershell
dotnet run --project src/RLSharp.Samples.Console/RLSharp.Samples.Console.csproj
```

FrozenLake web visualizer:

```powershell
dotnet run --project src/RLSharp.FrozenLake.Web/RLSharp.FrozenLake.Web.csproj
```

Default development URL:

```text
http://localhost:5111
```

## Common Agent APIs

All concrete agents inherit from `Agent`.

| API | Description |
| --- | --- |
| `Learn()` | Executes one algorithm-specific learning operation and returns `LearnOutcome`. |
| `RunEpisode()` | Runs one episode with the current policy. |
| `RunEpisodes(count)` | Runs multiple episodes. |
| `TestEpisodes(count)` | Returns average reward over multiple episodes. |
| `GetPolicyAct(state)` | Selects an action using the current policy. |
| `GetEpsilonAct(state)` | Selects an action using epsilon-greedy policy. |
| `Save(path)` | Saves model or value data. |
| `Load(path)` | Loads model or value data. |

Core environment types currently use these public names:

- `EnvironmentBase<TActionSpace, TObservationSpace>`
- `ActionValue`
- `ObservationValue`
- `Reward`
- `Step`
- `ReplayBuffer`
- `TrainerCallback`

## Testing

Run the default fast test suite:

```powershell
dotnet test src/RLSharp.Tests/RLSharp.Tests.csproj
```

The default test filter excludes long-running training and stochastic convergence
scenarios. To run all tests explicitly:

```powershell
dotnet test src/RLSharp.Tests/RLSharp.Tests.csproj `
  --filter "FullyQualifiedName~RLSharp.Tests" `
  --logger "console;verbosity=normal"
```

## Development Notes

- NuGet package versions are managed in `Directory.Packages.props`.
- CPU is the default TorchSharp runtime.
- Call `RandomProvider.SetSeed(seed)` at experiment entry points when reproducibility matters.
- Dispose temporary TorchSharp tensors directly or with `using` when extending algorithms.
- New algorithms should include deterministic model tests plus separate long-running convergence tests.

## License

RLSharp is available under the MIT License.

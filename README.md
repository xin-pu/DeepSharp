# DeepSharp

DeepSharp is a .NET 8 reinforcement-learning library built on
[TorchSharp](https://github.com/dotnet/TorchSharp). It includes tabular and
neural-network agents, reusable environments, experience replay, tests, a
console sample, and a FrozenLake web visualizer.

## Requirements

- .NET 8 SDK
- Windows, Linux, or macOS supported by the TorchSharp CPU package
- No CUDA installation is required by the default package configuration

## Build and test

```powershell
dotnet restore src/DeepSharp.sln
dotnet build src/DeepSharp.sln --no-restore
dotnet test src/TorchSharpTest/TorchSharpTest.csproj --no-build
```

Training and convergence tests can take substantially longer than model and
environment unit tests. CI runs a fast subset on every change and the complete
suite for pull requests.

## Minimal example

```csharp
using DeepSharp.RL;
using DeepSharp.RL.Agents.Tabular;
using DeepSharp.RL.Environs;

RandomProvider.SetSeed(42);

var environment = new FrozenLake([0.8f, 0.1f, 0.1f]);
var agent = new QLearning(
    environment,
    epsilon: 0.2f,
    alpha: 0.2f,
    gamma: 0.9f);

for (var episode = 0; episode < 100; episode++)
    agent.Learn();

var result = agent.RunEpisode();
Console.WriteLine($"Reward: {result.SumReward.Value}");
```

## Supported algorithms

| Family | Algorithms |
| --- | --- |
| Tabular | Q-Learning, SARSA, Monte Carlo, policy iteration, value iteration |
| Value-based deep RL | DQN, Double DQN, Dueling DQN, Noisy DQN, Categorical DQN, CGP |
| Policy-based deep RL | REINFORCE, cross entropy |
| Actor-critic | Actor-Critic, A2C, A3C |

## FrozenLake web demo

```powershell
dotnet run --project src/DeepSharp.FLWeb/DeepSharp.FLWeb.csproj
```

Open the URL printed by ASP.NET Core. Each SignalR connection receives only its
own training events. Disconnecting the client cancels its active training task.

## Project layout

- `src/DeepSharp.RL` — agents, environments, action selectors and replay buffers
- `src/DeepSharp.Core` — common trainer abstractions
- `src/DeepSharp.FLWeb` — FrozenLake SignalR web visualizer
- `src/RLConsole` — console entry point
- `src/TorchSharpTest` — xUnit test suite

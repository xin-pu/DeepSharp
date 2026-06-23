# DeepSharp

DeepSharp is a reinforcement learning project built with
[.NET 8](https://dotnet.microsoft.com/) and
[TorchSharp](https://github.com/dotnet/TorchSharp).

The project currently provides:

- Tabular reinforcement learning algorithms
- Neural-network-based deep reinforcement learning algorithms
- FrozenLake and K-Armed Bandit environments
- Reusable action selectors, experience replay buffers, and trainers
- A web-based FrozenLake visualizer
- An xUnit test suite and a console example

> DeepSharp is currently intended for learning, experimentation, and further
> development. It has not yet been published as a stable NuGet package, and its
> public APIs may still change.

## Requirements

- .NET 8 SDK
- Windows, Linux, or macOS
- No CUDA installation is required; the project uses `TorchSharp-cpu` by
  default

Check the installed .NET SDK:

```powershell
dotnet --version
```

## Project Structure

```text
DeepSharp
├─ src
│  ├─ DeepSharp.Core       Common trainer abstractions
│  ├─ DeepSharp.RL         Reinforcement learning library
│  ├─ DeepSharp.Utility    Tensor and utility helpers
│  ├─ DeepSharp.FLWeb      FrozenLake web visualizer
│  ├─ RLConsole            Console example
│  ├─ TorchSharpTest       xUnit test suite
│  └─ DeepSharp.sln
├─ images                  Documentation images
└─ resources               Additional resources
```

The main components of `DeepSharp.RL` are:

| Component | Description |
| --- | --- |
| `Agents` | Agent base classes and reinforcement learning algorithms |
| `Environs` | Environments, observations, actions, rewards, and spaces |
| `ActionSelectors` | Argmax, probability sampling, and epsilon-greedy selectors |
| `ExpReplays` | Uniform, prioritized, and episode replay buffers |
| `Trainers` | Training workflows, options, and callbacks |

## Clone and Build

```powershell
git clone https://github.com/xin-pu/DeepSharp.git
cd DeepSharp

dotnet restore src/DeepSharp.sln
dotnet build src/DeepSharp.sln --no-restore
```

After the solution builds successfully, you can run either the console example
or the web application.

## Quick Start: Q-Learning

The following example trains a Q-Learning agent on FrozenLake:

```csharp
using DeepSharp.RL;
using DeepSharp.RL.Agents.Tabular;
using DeepSharp.RL.Environs;

// Set both the managed and TorchSharp random seeds.
RandomProvider.SetSeed(42);

// The probabilities represent moving toward the selected direction,
// slipping left, and slipping right. They must add up to 1.
var environment = new FrozenLake([0.8f, 0.1f, 0.1f]);

var agent = new QLearning(
    environment,
    epsilon: 0.2f,
    alpha: 0.2f,
    gamma: 0.9f);

for (var episode = 0; episode < 1_000; episode++)
{
    var outcome = agent.Learn();

    if ((episode + 1) % 100 == 0)
    {
        var averageReward = agent.TestEpisodes(20);
        Console.WriteLine(
            $"Episode: {episode + 1}, Average reward: {averageReward:F3}");
    }
}

// Run one complete episode with the learned policy.
var result = agent.RunEpisode();
Console.WriteLine($"Reward: {result.SumReward.Value}");
Console.WriteLine(result);
```

The basic interaction model is:

```text
Environment.Reset()
        ↓
Agent selects an Act
        ↓
Environment.Step(Act)
        ↓
Returns Step:
PreState, Action, PostState, Reward, IsComplete
        ↓
Agent updates its policy or value function
```

The amount of work performed by `Agent.Learn()` depends on the algorithm.
Q-Learning processes one episode per call, while DQN may collect and train on
multiple episodes according to its constructor parameters.

## Using DQN

```csharp
using DeepSharp.RL;
using DeepSharp.RL.Agents.Deep.Value;
using DeepSharp.RL.Environs;

RandomProvider.SetSeed(42);

var environment = new FrozenLake([0.8f, 0.1f, 0.1f]);

var agent = new DQN(
    environment,
    n: 100,
    c: 1_000,
    epsilon: 0.2f,
    gamma: 0.99f,
    batchSize: 32);

for (var iteration = 0; iteration < 100; iteration++)
{
    var outcome = agent.Learn();
    var averageReward = agent.TestEpisodes(20);

    Console.WriteLine(
        $"Iteration: {iteration + 1}, " +
        $"Loss: {outcome.Evaluate:F4}, " +
        $"Reward: {averageReward:F3}");
}
```

DQN parameters:

| Parameter | Description |
| --- | --- |
| `n` | Number of episodes processed by one `Learn()` call and target-network synchronization interval |
| `c` | Experience replay capacity |
| `epsilon` | Random exploration probability |
| `gamma` | Reward discount factor |
| `batchSize` | Network update batch size; it must not exceed `c` |

The project uses the CPU TorchSharp runtime by default. The DQN networks follow
the device used by the environment. To use CUDA, replace the TorchSharp runtime
package and explicitly create the environment with a CUDA device.

## Included Algorithms

| Category | Implementations |
| --- | --- |
| Tabular methods | Q-Learning, SARSA, on-policy and off-policy Monte Carlo |
| Dynamic programming | Policy Iteration, Value Iteration, and related variants |
| Value-based deep RL | DQN, Double DQN, Dueling DQN, Noisy DQN, Categorical DQN, CGP |
| Policy-based deep RL | REINFORCE and Cross Entropy |
| Actor-Critic | Actor-Critic, A2C, and A3C |

The web interface currently supports these agents:

- `QLearning`
- `SARSA`
- `MonteCarloOnPolicy`
- `MonteCarloOffPolicy`
- `DQN`
- `REINFORCE`
- `A2C`

Other algorithms can be used directly through the `DeepSharp.RL` API, but they
have not yet been added to the web application's agent factory.

## Running the Console Example

`RLConsole` currently demonstrates DQN training on FrozenLake:

```powershell
dotnet run --project src/RLConsole/RLConsole.csproj
```

The example continues training until its average test reward reaches the
threshold defined in the source code. Runtime is not fixed because learning
results depend on randomness and hyperparameters.

## Running the FrozenLake Web Application

```powershell
dotnet run --project src/DeepSharp.FLWeb/DeepSharp.FLWeb.csproj
```

The default development URL is:

```text
http://localhost:5111
```

The web application uses ASP.NET Core, SignalR, and browser-side JavaScript. It
allows you to:

- Select an agent
- Configure learning rate, discount factor, epsilon, and replay parameters
- Start and stop training
- Observe FrozenLake state changes and episode metrics in real time
- Run a demonstration with the trained agent

The three FrozenLake movement probabilities must be within `[0, 1]` and must
add up to `1`. Replay capacity, batch size, episode limits, and other values are
also validated by the server.

Training tasks are isolated by SignalR connection. Disconnecting the browser
cancels the corresponding training task.

## Common Agent APIs

All concrete agents inherit from `Agent`.

| API | Description |
| --- | --- |
| `Learn()` | Executes one algorithm-specific learning operation and returns a `LearnOutcome` |
| `RunEpisode()` | Runs one episode with the current policy |
| `RunEpisodes(count)` | Runs multiple episodes |
| `TestEpisodes(count)` | Returns the average reward over multiple episodes |
| `GetPolicyAct(state)` | Selects an action using the current policy |
| `GetEpsilonAct(state)` | Selects an action using an epsilon-greedy policy |
| `Save(path)` | Saves model or value data |
| `Load(path)` | Loads previously saved data |

`LearnOutcome` contains the steps produced by the learning operation and an
evaluation value. Each `Step` contains:

- `PreState`
- `Action`
- `PostState`
- `Reward`
- `IsComplete`
- `Priority`

## Creating a Custom Environment

A custom environment should inherit from:

```csharp
Environ<TActionSpace, TObservationSpace>
```

At minimum, it must implement:

```csharp
public override Observation Update(Act act);
public override Reward GetReward(Observation observation);
public override bool IsComplete(int epoch);
public override float GetReturn(Episode episode);
```

The constructor must also initialize the action and observation spaces:

```csharp
ActionSpace = ...;
ObservationSpace = ...;
Reset();
```

The following space types are included:

- `Discrete`
- `Binary`
- `MultiDiscrete`
- `MultiBinary`
- `Box`

Environments communicate with agents through `Step`, so a custom environment
normally does not need to depend on a specific learning algorithm.

## Using `RLTrainer`

Use `RLTrainer` when you need a consistent training and validation workflow,
callbacks, cancellation, or automatic checkpoint saving:

```csharp
using DeepSharp.RL.Agents.Tabular;
using DeepSharp.RL.Environs;
using DeepSharp.RL.Trainers;

var environment = new FrozenLake([0.8f, 0.1f, 0.1f]);
var agent = new QLearning(environment);
var trainer = new RLTrainer(agent, Console.WriteLine);

using var cancellation = new CancellationTokenSource(
    TimeSpan.FromMinutes(5));

trainer.Train(
    preReward: 0.8f,
    trainEpoch: 1_000,
    testEpisodes: 20,
    testInterval: 10,
    cancellationToken: cancellation.Token);
```

The trainer stops when the validation average reaches `preReward`. Training can
also be stopped through the supplied `CancellationToken`.

## Testing

Run the default fast test suite:

```powershell
dotnet test src/TorchSharpTest/TorchSharpTest.csproj
```

The default test filter excludes long-running training and stochastic
convergence scenarios. It is intended for regular development and CI checks.

Run the complete test suite explicitly:

```powershell
dotnet test src/TorchSharpTest/TorchSharpTest.csproj `
  --filter "FullyQualifiedName~TorchSharpTest" `
  --logger "console;verbosity=normal"
```

The complete suite includes deep reinforcement learning and convergence tests.
It may require several minutes or longer, depending on machine performance and
randomness.

## Development Notes

- NuGet package versions are managed in `Directory.Packages.props`.
- CPU is the default device.
- Call `RandomProvider.SetSeed(seed)` at experiment entry points when
  reproducibility matters.
- TorchSharp tensors own native resources. Dispose temporary tensors directly
  or manage them with `using` when extending the algorithms.
- Replay sampling returns an `ExperienceCase` that implements `IDisposable` and
  should be disposed after use.
- New algorithms should include deterministic model tests and separate
  long-running convergence tests.

## License

DeepSharp is available under the [MIT License](LICENSE).

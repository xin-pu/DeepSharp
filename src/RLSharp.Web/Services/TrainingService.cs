using System.Threading.Channels;
using Microsoft.AspNetCore.SignalR;
using RLSharp.Core.Agents;
using RLSharp.Torch.Agents;
using RLSharp.Torch.Agents.Generic;
using RLSharp.Torch.Encoding;
using RLSharp.Torch.Environs;
using RLSharp.Torch.Examples.CartPole;
using RLSharp.Torch.Examples.RiskyBandit;
using RLSharp.Web.Hub;
using RLSharp.Web.Models;

namespace RLSharp.Web.Services
{
	public sealed class TrainingService : ITrainingService, IAsyncDisposable
	{
		private readonly CancellationTokenSource _disposeCts = new();
		private readonly Task                    _eventPump;

		private readonly Channel<ClientEvent> _events = Channel.CreateBounded<ClientEvent>(
			new BoundedChannelOptions(256)
			{
				FullMode     = BoundedChannelFullMode.DropOldest,
				SingleReader = true,
				SingleWriter = false
			});

		private readonly IHubContext<TrainingHub> _hubContext;
		private readonly SemaphoreSlim            _lifecycle = new(1, 1);
		private readonly ILogger<TrainingService> _logger;

		private Agent?                                       _agent;
		private IAgent<RiskyBanditState, RiskyBanditAction>? _banditAgent;
		private RiskyBanditEnvironment?                      _banditEnv;
		private IAgent<CartPoleState, CartPoleAction>?       _cartPoleAgent;
		private CartPoleEnvironment?                         _cartPoleEnv;
		private FrozenLake?                                  _env;
		private string                                       _environmentType = "FrozenLake";
		private int                                          _episodeCount;
		private string?                                      _ownerConnectionId;
		private CancellationTokenSource?                     _runCts;
		private Task?                                        _runTask;

		public TrainingService(IHubContext<TrainingHub> hubContext, ILogger<TrainingService> logger)
		{
			_hubContext = hubContext;
			_logger     = logger;
			_eventPump  = PumpEventsAsync(_disposeCts.Token);
		}

		public async ValueTask DisposeAsync()
		{
			_runCts?.Cancel();
			if (_runTask != null)
				await _runTask;
			_events.Writer.TryComplete();
			_disposeCts.Cancel();
			try
			{
				await _eventPump;
			}
			catch (OperationCanceledException)
			{
			}

			DisposeRunToken();
			_disposeCts.Dispose();
			_lifecycle.Dispose();
		}

		public bool IsTraining => _runTask is { IsCompleted: false };

		public async Task StartTraining(TrainingConfig config, string connectionId)
		{
			ArgumentNullException.ThrowIfNull(config);
			config.Validate();

			await _lifecycle.WaitAsync();
			try
			{
				if (IsTraining)
				{
					Queue(connectionId, "Error", "Training or demo is already running.");
					return;
				}

				DisposeRunToken();
				_environmentType = config.EnvironmentType;
				if (_environmentType == "CartPole")
				{
					CreateCartPole(config);
				}
				else if (_environmentType == "RiskyBandit")
				{
					CreateRiskyBandit(config);
				}
				else
				{
					_env   = new FrozenLake([config.SmoothTarget, config.SmoothLeft, config.SmoothRight]);
					_agent = AgentFactory.Create(config, _env);
					AttachStepCallback(connectionId, _runCts?.Token ?? CancellationToken.None);
				}

				_ownerConnectionId = connectionId;
				_episodeCount      = 0;
				_runCts            = new CancellationTokenSource();
				if (_environmentType == "FrozenLake")
					AttachStepCallback(connectionId, _runCts.Token);
				Queue(connectionId, "TrainingStarted", $"{config.EnvironmentType}-{config.AgentType}");
				_runTask = RunTrainingAsync(config, connectionId, _runCts.Token);
			}
			catch (Exception ex)
			{
				_logger.LogError(ex, "Failed to start training");
				Queue(connectionId, "Error", $"Failed to start training: {ex.Message}");
				DisposeRunToken();
				_ownerConnectionId = null;
			}
			finally
			{
				_lifecycle.Release();
			}
		}

		public async Task StopTraining(string connectionId)
		{
			Task? task = null;
			await _lifecycle.WaitAsync();
			try
			{
				if (_ownerConnectionId != connectionId)
					return;

				_runCts?.Cancel();
				task = _runTask;
			}
			finally
			{
				_lifecycle.Release();
			}

			if (task != null)
				try
				{
					await task;
				}
				catch (OperationCanceledException)
				{
					// Expected during stop.
				}

			await _lifecycle.WaitAsync();
			try
			{
				if (_runTask == task)
				{
					_runTask           = null;
					_ownerConnectionId = null;
					DisposeRunToken();
				}
			}
			finally
			{
				_lifecycle.Release();
			}
		}

		public async Task RunDemo(string connectionId)
		{
			await _lifecycle.WaitAsync();
			try
			{
				if (IsTraining)
				{
					Queue(connectionId, "Error", "Training or demo is already running.");
					return;
				}

				if (_environmentType == "CartPole")
				{
					if (_cartPoleAgent == null || _cartPoleEnv == null)
					{
						Queue(connectionId, "Error", "No trained CartPole agent. Please train first.");
						return;
					}
				}
				else if (_environmentType == "RiskyBandit")
				{
					if (_banditAgent == null || _banditEnv == null)
					{
						Queue(connectionId, "Error", "No trained RiskyBandit agent. Please train first.");
						return;
					}
				}
				else if (_agent == null || _env == null)
				{
					Queue(connectionId, "Error", "No trained agent. Please train first.");
					return;
				}

				DisposeRunToken();
				_ownerConnectionId = connectionId;
				_runCts            = new CancellationTokenSource();
				if (_environmentType == "FrozenLake")
					AttachStepCallback(connectionId, _runCts.Token);
				Queue(connectionId, "TrainingStarted", _environmentType switch
				{
					"CartPole"    => "Demo-CartPole",
					"RiskyBandit" => "Demo-RiskyBandit",
					_             => $"Demo-{_agent?.Name}"
				});
				_runTask = RunDemoAsync(connectionId, _runCts.Token);
			}
			finally
			{
				_lifecycle.Release();
			}
		}

		public Task ResetEnvironment(string connectionId)
		{
			if (IsTraining || _ownerConnectionId != connectionId)
				return Task.CompletedTask;

			if (_environmentType == "CartPole")
			{
				if (_cartPoleEnv == null)
					return Task.CompletedTask;
				var state = _cartPoleEnv.Reset();
				Queue(connectionId, "StepUpdate", CreateCartPoleVisualState(state, null, 0, false));
				return Task.CompletedTask;
			}

			if (_environmentType == "RiskyBandit")
			{
				_banditEnv ??= new RiskyBanditEnvironment(42);
				var state = _banditEnv.Reset();
				Queue(connectionId, "StepUpdate", CreateRiskyBanditVisualState(state, null, 0, false));
				return Task.CompletedTask;
			}

			if (_env == null)
				return Task.CompletedTask;

			_env.Reset();
			var gridState = GridStateExtractor.Extract(_env, new Step(
				_env.ObservationValue!,
				new ActionValue(torch.tensor(0)),
				_env.ObservationValue!,
				new Reward(0)));
			Queue(connectionId, "StepUpdate", gridState);
			return Task.CompletedTask;
		}

		public Task ManualStep(string environmentType, string action, string connectionId)
		{
			if (IsTraining)
				return Task.CompletedTask;

			_environmentType   =   string.IsNullOrWhiteSpace(environmentType) ? _environmentType : environmentType;
			_ownerConnectionId ??= connectionId;

			if (_environmentType == "CartPole")
			{
				_cartPoleEnv ??= new CartPoleEnvironment(42);
				if (!Enum.TryParse<CartPoleAction>(action, true, out var cartAction))
					cartAction = CartPoleAction.Right;
				var result = _cartPoleEnv.Step(cartAction);
				Queue(connectionId, "StepUpdate",
					CreateCartPoleVisualState(result.State, cartAction, result.Reward, result.IsTerminal));
				if (result.IsTerminal)
					_cartPoleEnv.Reset();
				return Task.CompletedTask;
			}

			if (_environmentType == "RiskyBandit")
			{
				_banditEnv ??= new RiskyBanditEnvironment(42);
				if (!Enum.TryParse<RiskyBanditAction>(action, true, out var banditAction))
					banditAction = RiskyBanditAction.Safe;
				var result = _banditEnv.Step(banditAction);
				Queue(connectionId, "StepUpdate",
					CreateRiskyBanditVisualState(result.State, banditAction, result.Reward, result.IsTerminal));
				if (result.IsTerminal)
					_banditEnv.Reset();
				return Task.CompletedTask;
			}

			_env ??= new FrozenLake([0.8f, 0.1f, 0.1f]);
			var actionIndex = action.ToLowerInvariant() switch
			{
				"left"  => 0,
				"down"  => 1,
				"right" => 2,
				"up"    => 3,
				_       => 2
			};
			_env.ObservationValue ??= _env.Reset();
			var step = _env.Step(new ActionValue(torch.tensor(actionIndex)), _env.Life);
			Queue(connectionId, "StepUpdate", GridStateExtractor.Extract(_env, step));
			if (step.IsComplete)
				_env.Reset();
			return Task.CompletedTask;
		}

		private async Task RunTrainingAsync(TrainingConfig config, string connectionId, CancellationToken token)
		{
			try
			{
				while (!token.IsCancellationRequested && _episodeCount < config.MaxEpisodes)
				{
					if (_environmentType == "CartPole")
					{
						await RunCartPoleTrainingEpisodeAsync(config, connectionId, token);
						continue;
					}

					if (_environmentType == "RiskyBandit")
					{
						await RunRiskyBanditTrainingEpisodeAsync(config, connectionId, token);
						continue;
					}

					var agent = _agent ?? throw new InvalidOperationException("Agent is unavailable.");
					_env?.Reset();
					var outcome = agent.Learn();
					var count   = Interlocked.Increment(ref _episodeCount);
					Queue(connectionId, "EpisodeEnd",
						CreateProgress(count, outcome.Steps, outcome.Evaluate, agent.Epsilon));
					if (config.SpeedDelayMs > 0)
						await Task.Delay(config.SpeedDelayMs, token);
				}
			}
			catch (OperationCanceledException) when (token.IsCancellationRequested)
			{
			}
			catch (Exception ex)
			{
				_logger.LogError(ex, "Training error");
				Queue(connectionId, "Error", ex.Message);
			}
			finally
			{
				Queue(connectionId, "TrainingStopped", null);
			}
		}

		private async Task RunDemoAsync(string connectionId, CancellationToken token)
		{
			try
			{
				for (var i = 0; i < 10; i++)
				{
					token.ThrowIfCancellationRequested();
					if (_environmentType == "CartPole")
					{
						await RunCartPoleVisualEpisodeAsync(connectionId, token, 800);
						Queue(connectionId, "EpisodeEnd", new TrainingProgress
						{
							EpisodeCount  = i + 1,
							StepCount     = 0,
							SumReward     = 0,
							AverageReward = 0,
							Epsilon       = 0,
							Loss          = 0
						});
						continue;
					}

					if (_environmentType == "RiskyBandit")
					{
						await RunRiskyBanditVisualEpisodeAsync(connectionId, token, 800);
						Queue(connectionId, "EpisodeEnd", new TrainingProgress
						{
							EpisodeCount  = i + 1,
							StepCount     = 0,
							SumReward     = 0,
							AverageReward = 0,
							Epsilon = _banditAgent is QLearningAgent<RiskyBanditState, RiskyBanditAction> q
								? q.Epsilon
								: 0,
							Loss = 0
						});
						continue;
					}

					var agent = _agent ?? throw new InvalidOperationException("Agent is unavailable.");
					_env?.Reset();
					var episode = agent.RunEpisode();
					Queue(connectionId, "EpisodeEnd", CreateProgress(i + 1, episode.Steps, 0, agent.Epsilon));
					await Task.Delay(800, token);
				}
			}
			catch (OperationCanceledException) when (token.IsCancellationRequested)
			{
			}
			catch (Exception ex)
			{
				_logger.LogError(ex, "Demo error");
				Queue(connectionId, "Error", ex.Message);
			}
			finally
			{
				Queue(connectionId, "TrainingStopped", null);
			}
		}

		private void AttachStepCallback(string connectionId, CancellationToken token)
		{
			if (_env == null)
				return;

			_env.CallBack = step =>
			{
				if (!token.IsCancellationRequested && _env != null)
					Queue(connectionId, "StepUpdate", GridStateExtractor.Extract(_env, step));
			};
		}

		private void CreateCartPole(TrainingConfig config)
		{
			_env         = null;
			_agent       = null;
			_banditEnv   = null;
			_banditAgent = null;
			_cartPoleEnv = new CartPoleEnvironment(42);
			var encoder = new DelegateStateEncoder<CartPoleState>(4, state => state.ToFeatures());
			_cartPoleAgent = config.AgentType == "PPO"
				? new PpoAgent<CartPoleState, CartPoleAction>(
					_cartPoleEnv,
					encoder,
					gamma: config.Gamma,
					learningRate: config.Alpha,
					maxStepsPerEpisode: 500)
				: new DqnAgent<CartPoleState, CartPoleAction>(
					_cartPoleEnv,
					encoder,
					epsilon: config.Epsilon,
					gamma: config.Gamma,
					learningRate: config.Alpha,
					maxStepsPerEpisode: 500,
					seed: 42);
		}

		private void CreateRiskyBandit(TrainingConfig config)
		{
			_env           = null;
			_agent         = null;
			_cartPoleEnv   = null;
			_cartPoleAgent = null;
			_banditEnv     = new RiskyBanditEnvironment(42);
			_banditAgent = new QLearningAgent<RiskyBanditState, RiskyBanditAction>(
				_banditEnv,
				_ => "bandit",
				config.Epsilon,
				config.Alpha,
				config.Gamma,
				100,
				42);
		}

		private async Task RunCartPoleTrainingEpisodeAsync(
			TrainingConfig    config,
			string            connectionId,
			CancellationToken token)
		{
			var agent  = _cartPoleAgent ?? throw new InvalidOperationException("CartPole agent is unavailable.");
			var result = agent.Learn();
			var count  = Interlocked.Increment(ref _episodeCount);

			Queue(connectionId, "EpisodeEnd", new TrainingProgress
			{
				EpisodeCount  = count,
				StepCount     = result.Steps,
				SumReward     = result.Reward,
				AverageReward = result.Steps == 0 ? 0 : result.Reward / result.Steps,
				Epsilon       = agent is DqnAgent<CartPoleState, CartPoleAction> dqn ? dqn.Epsilon : 0,
				Loss          = result.Loss ?? 0
			});

			await RunCartPoleVisualEpisodeAsync(connectionId, token, Math.Max(10, config.SpeedDelayMs));
		}

		private async Task RunCartPoleVisualEpisodeAsync(string connectionId, CancellationToken token, int delayMs)
		{
			var env   = _cartPoleEnv   ?? throw new InvalidOperationException("CartPole environment is unavailable.");
			var agent = _cartPoleAgent ?? throw new InvalidOperationException("CartPole agent is unavailable.");
			var state = env.Reset();
			Queue(connectionId, "StepUpdate", CreateCartPoleVisualState(state, null, 0, false));

			for (var step = 0; step < 500; step++)
			{
				token.ThrowIfCancellationRequested();
				var action = agent.SelectAction(state);
				var result = env.Step(action);
				Queue(connectionId, "StepUpdate",
					CreateCartPoleVisualState(result.State, action, result.Reward, result.IsTerminal));
				state = result.State;
				await Task.Delay(delayMs, token);
				if (result.IsTerminal)
					break;
			}
		}

		private async Task RunRiskyBanditTrainingEpisodeAsync(
			TrainingConfig    config,
			string            connectionId,
			CancellationToken token)
		{
			var agent  = _banditAgent ?? throw new InvalidOperationException("RiskyBandit agent is unavailable.");
			var result = agent.Learn();
			var count  = Interlocked.Increment(ref _episodeCount);

			Queue(connectionId, "EpisodeEnd", new TrainingProgress
			{
				EpisodeCount  = count,
				StepCount     = result.Steps,
				SumReward     = result.Reward,
				AverageReward = result.Steps == 0 ? 0 : result.Reward / result.Steps,
				Epsilon       = agent is QLearningAgent<RiskyBanditState, RiskyBanditAction> q ? q.Epsilon : 0,
				Loss          = result.Loss ?? 0
			});

			await RunRiskyBanditVisualEpisodeAsync(connectionId, token, Math.Max(10, config.SpeedDelayMs));
		}

		private async Task RunRiskyBanditVisualEpisodeAsync(string connectionId, CancellationToken token, int delayMs)
		{
			var env   = _banditEnv   ?? throw new InvalidOperationException("RiskyBandit environment is unavailable.");
			var agent = _banditAgent ?? throw new InvalidOperationException("RiskyBandit agent is unavailable.");
			var state = env.Reset();
			Queue(connectionId, "StepUpdate", CreateRiskyBanditVisualState(state, null, 0, false));

			for (var step = 0; step < env.MaxSteps; step++)
			{
				token.ThrowIfCancellationRequested();
				var action = agent.SelectAction(state);
				var result = env.Step(action);
				Queue(connectionId, "StepUpdate",
					CreateRiskyBanditVisualState(result.State, action, result.Reward, result.IsTerminal));
				state = result.State;
				await Task.Delay(delayMs, token);
				if (result.IsTerminal)
					break;
			}
		}

		private static CartPoleVisualState CreateCartPoleVisualState(
			CartPoleState   state,
			CartPoleAction? action,
			float           reward,
			bool            isComplete)
		{
			return new CartPoleVisualState
			{
				Position        = state.Position,
				Velocity        = state.Velocity,
				Angle           = state.Angle,
				AngularVelocity = state.AngularVelocity,
				ActionName      = action?.ToString() ?? "-",
				Reward          = reward,
				IsComplete      = isComplete
			};
		}

		private static RiskyBanditVisualState CreateRiskyBanditVisualState(
			RiskyBanditState   state,
			RiskyBanditAction? action,
			float              reward,
			bool               isComplete)
		{
			return new RiskyBanditVisualState
			{
				Step        = state.Step,
				LastAction  = action.HasValue ? (int)action.Value : state.LastAction,
				ActionName  = action?.ToString() ?? "-",
				Reward      = reward,
				TotalReward = state.TotalReward,
				IsComplete  = isComplete
			};
		}

		private static TrainingProgress CreateProgress(int count,
			IReadOnlyCollection<Step>                      steps,
			float                                          loss,
			float                                          epsilon)
		{
			return new TrainingProgress
			{
				EpisodeCount  = count,
				StepCount     = steps.Count,
				SumReward     = steps.Sum(step => step.Reward.Value),
				AverageReward = steps.Count == 0 ? 0 : steps.Average(step => step.Reward.Value),
				Epsilon       = epsilon,
				Loss          = loss
			};
		}

		private void Queue(string connectionId, string method, object? payload)
		{
			if (!_events.Writer.TryWrite(new ClientEvent(connectionId, method, payload)))
				_logger.LogWarning("Dropped SignalR event {Method} for {ConnectionId}", method, connectionId);
		}

		private async Task PumpEventsAsync(CancellationToken token)
		{
			await foreach (var message in _events.Reader.ReadAllAsync(token))
				try
				{
					if (message.Payload == null)
						await _hubContext.Clients.Client(message.ConnectionId).SendAsync(message.Method, token);
					else
						await _hubContext.Clients.Client(message.ConnectionId)
							.SendAsync(message.Method, message.Payload, token);
				}
				catch (OperationCanceledException) when (token.IsCancellationRequested)
				{
					break;
				}
				catch (Exception ex)
				{
					_logger.LogWarning(ex, "Failed to send SignalR event {Method}", message.Method);
				}
		}

		private void DisposeRunToken()
		{
			_runCts?.Dispose();
			_runCts = null;
		}

		private sealed record ClientEvent(string ConnectionId, string Method, object? Payload);
	}
}
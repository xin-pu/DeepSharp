using System.Threading.Channels;
using Microsoft.AspNetCore.SignalR;
using RLSharp.FrozenLake.Web.Hub;
using RLSharp.FrozenLake.Web.Models;
using RLSharp.Torch.Agents;
using RLSharp.Torch.Environs;

namespace RLSharp.FrozenLake.Web.Services
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

		private Agent?                     _agent;
		private Torch.Environs.FrozenLake? _env;
		private int                        _episodeCount;
		private string?                    _ownerConnectionId;
		private CancellationTokenSource?   _runCts;
		private Task?                      _runTask;

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
				_env = new Torch.Environs.FrozenLake([config.SmoothTarget, config.SmoothLeft, config.SmoothRight]);
				_agent = AgentFactory.Create(config, _env);
				_ownerConnectionId = connectionId;
				_episodeCount = 0;
				_runCts = new CancellationTokenSource();
				AttachStepCallback(connectionId, _runCts.Token);
				Queue(connectionId, "TrainingStarted", config.AgentType);
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

				if (_agent == null || _env == null)
				{
					Queue(connectionId, "Error", "No trained agent. Please train first.");
					return;
				}

				DisposeRunToken();
				_ownerConnectionId = connectionId;
				_runCts            = new CancellationTokenSource();
				AttachStepCallback(connectionId, _runCts.Token);
				Queue(connectionId, "TrainingStarted", $"Demo-{_agent.Name}");
				_runTask = RunDemoAsync(connectionId, _runCts.Token);
			}
			finally
			{
				_lifecycle.Release();
			}
		}

		public Task ResetEnvironment(string connectionId)
		{
			if (_env == null || IsTraining || _ownerConnectionId != connectionId)
				return Task.CompletedTask;

			_env.Reset();
			var state = GridStateExtractor.Extract(_env, new Step(
				_env.ObservationValue!,
				new ActionValue(torch.tensor(0)),
				_env.ObservationValue!,
				new Reward(0)));
			Queue(connectionId, "StepUpdate", state);
			return Task.CompletedTask;
		}

		private async Task RunTrainingAsync(TrainingConfig config, string connectionId, CancellationToken token)
		{
			try
			{
				while (!token.IsCancellationRequested && _episodeCount < config.MaxEpisodes)
				{
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
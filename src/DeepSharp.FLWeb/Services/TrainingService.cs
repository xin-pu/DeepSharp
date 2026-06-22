using DeepSharp.FLWeb.Hub;
using DeepSharp.FLWeb.Models;
using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using Microsoft.AspNetCore.SignalR;

namespace DeepSharp.FLWeb.Services;

public class TrainingService : ITrainingService
{
    private readonly IHubContext<TrainingHub> _hubContext;
    private readonly ILogger<TrainingService> _logger;
    private CancellationTokenSource? _cts;
    private FrozenLake? _env;
    private Agent? _agent;
    private int _episodeCount;
    private volatile bool _isTraining;
    private volatile bool _isDemoRunning;

    public bool IsTraining => _isTraining || _isDemoRunning;

    public TrainingService(IHubContext<TrainingHub> hubContext, ILogger<TrainingService> logger)
    {
        _hubContext = hubContext;
        _logger = logger;
    }

    public async Task StartTraining(TrainingConfig config, string connectionId)
    {
        if (_isTraining || _isDemoRunning)
        {
            await _hubContext.Clients.Client(connectionId)
                .SendAsync("Error", "Training or demo is already running.");
            return;
        }

        _cts = new CancellationTokenSource();
        _isTraining = true;
        _episodeCount = 0;

        // Create environment
        _env = new FrozenLake(new[] { config.SmoothTarget, config.SmoothLeft, config.SmoothRight });
        // Create agent
        _agent = AgentFactory.Create(config, _env);

        // Hook: push each step to SignalR
        _env.CallBack = step =>
        {
            if (_cts.IsCancellationRequested || _env == null) return;
            var gridState = GridStateExtractor.Extract(_env, step);
            _hubContext.Clients.All.SendAsync("StepUpdate", gridState);
        };

        await _hubContext.Clients.All.SendAsync("TrainingStarted", config.AgentType);

        // Run training on background thread
        _ = Task.Run(async () =>
        {
            try
            {
                while (!_cts.IsCancellationRequested && _episodeCount < config.MaxEpisodes)
                {
                    if (_agent == null || _env == null) break;

                    _env.Reset();
                    var outcome = _agent.Learn();
                    Interlocked.Increment(ref _episodeCount);

                    var progress = new TrainingProgress
                    {
                        EpisodeCount = _episodeCount,
                        StepCount = outcome.Steps.Count,
                        SumReward = outcome.Steps.Sum(s => s.Reward.Value),
                        AverageReward = outcome.Steps.Count > 0
                            ? outcome.Steps.Average(s => s.Reward.Value)
                            : 0,
                        Epsilon = _agent.Epsilon,
                        Loss = outcome.Evaluate
                    };

                    await _hubContext.Clients.All.SendAsync("EpisodeEnd", progress);

                    // Throttle between episodes for visualization
                    if (config.SpeedDelayMs > 0 && !_cts.IsCancellationRequested)
                    {
                        try
                        {
                            await Task.Delay(config.SpeedDelayMs, _cts.Token);
                        }
                        catch (OperationCanceledException)
                        {
                            break;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Training error");
                await _hubContext.Clients.All.SendAsync("Error", ex.Message);
            }
            finally
            {
                _isTraining = false;
                await _hubContext.Clients.All.SendAsync("TrainingStopped");
            }
        }, _cts.Token);
    }

    public async Task StopTraining()
    {
        if (_cts != null)
        {
            await _cts.CancelAsync();
            _cts.Dispose();
            _cts = null;
        }
    }

    public async Task RunDemo()
    {
        if (_isTraining || _isDemoRunning)
        {
            await _hubContext.Clients.All.SendAsync("Error", "Already running.");
            return;
        }

        if (_agent == null || _env == null)
        {
            await _hubContext.Clients.All.SendAsync("Error", "No trained agent. Please train first.");
            return;
        }

        _cts = new CancellationTokenSource();
        _isDemoRunning = true;

        // Hook for demo steps
        _env.CallBack = step =>
        {
            if (_cts.IsCancellationRequested || _env == null) return;
            var gridState = GridStateExtractor.Extract(_env, step);
            _hubContext.Clients.All.SendAsync("StepUpdate", gridState);
        };

        await _hubContext.Clients.All.SendAsync("TrainingStarted", $"Demo-{_agent.Name}");

        _ = Task.Run(async () =>
        {
            try
            {
                // Run 10 demo episodes
                for (var i = 0; i < 10 && !_cts.IsCancellationRequested; i++)
                {
                    _env.Reset();
                    var episode = _agent.RunEpisode();

                    var progress = new TrainingProgress
                    {
                        EpisodeCount = i + 1,
                        StepCount = episode.Steps.Count,
                        SumReward = episode.SumReward.Value,
                        AverageReward = episode.Steps.Count > 0
                            ? episode.Steps.Average(s => s.Reward.Value)
                            : 0,
                        Epsilon = _agent.Epsilon,
                        Loss = 0
                    };

                    await _hubContext.Clients.All.SendAsync("EpisodeEnd", progress);

                    if (!_cts.IsCancellationRequested)
                        await Task.Delay(800, _cts.Token);
                }
            }
            catch (OperationCanceledException)
            {
                // Normal cancellation
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Demo error");
                await _hubContext.Clients.All.SendAsync("Error", ex.Message);
            }
            finally
            {
                _isDemoRunning = false;
                await _hubContext.Clients.All.SendAsync("TrainingStopped");
            }
        }, _cts.Token);
    }

    public Task ResetEnvironment()
    {
        if (_env != null && !_isTraining && !_isDemoRunning)
        {
            _env.Reset();
            var gridState = GridStateExtractor.Extract(_env, new Step(
                _env.Observation!,
                new Act(torch.tensor(0)),
                _env.Observation!,
                new Reward(0),
                false));
            return _hubContext.Clients.All.SendAsync("StepUpdate", gridState);
        }
        return Task.CompletedTask;
    }
}

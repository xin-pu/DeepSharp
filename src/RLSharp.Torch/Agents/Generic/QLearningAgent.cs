using System.Text.Json;
using RLSharp.Core.Agents;
using RLSharp.Core.Environments;
using RLSharp.Core.Training;

namespace RLSharp.Torch.Agents.Generic
{
	public sealed class QLearningAgent<TState, TAction> : IAgent<TState, TAction>
	{
		private readonly IEnvironment<TState, TAction>     _environment;
		private readonly Dictionary<StateActionKey, float> _q = [];
		private readonly Random                            _random;
		private readonly Func<TState, string>              _stateKey;

		public QLearningAgent(
			IEnvironment<TState, TAction> environment,
			Func<TState, string>?         stateKey           = null,
			float                         epsilon            = 0.2f,
			float                         alpha              = 0.2f,
			float                         gamma              = 0.9f,
			int                           maxStepsPerEpisode = 1_000,
			int?                          seed               = null)
		{
			_environment       = environment ?? throw new ArgumentNullException(nameof(environment));
			_stateKey          = stateKey    ?? (state => JsonSerializer.Serialize(state));
			_random            = seed is null ? new Random() : new Random(seed.Value);
			Epsilon            = epsilon;
			Alpha              = alpha;
			Gamma              = gamma;
			MaxStepsPerEpisode = maxStepsPerEpisode;
		}

		public float Epsilon { get; set; }

		public float Alpha { get; set; }

		public float Gamma { get; set; }

		public int MaxStepsPerEpisode { get; set; }

		public LearnResult Learn()
		{
			var state       = _environment.Reset();
			var totalReward = 0f;
			var steps       = 0;

			while (steps < MaxStepsPerEpisode)
			{
				var action   = SelectExploratoryAction(state);
				var result   = _environment.Step(action);
				var oldValue = GetValue(state, action);
				var target   = result.Reward     + (result.IsTerminal ? 0 : Gamma * BestValue(result.State));
				SetValue(state, action, oldValue + Alpha * (target - oldValue));

				totalReward += result.Reward;
				steps++;
				state = result.State;
				if (result.IsTerminal)
					break;
			}

			return new LearnResult(steps, totalReward);
		}

		public TAction SelectAction(TState state)
		{
			var actions    = _environment.ActionSpace.Actions;
			var bestAction = actions[0];
			var bestValue  = GetValue(state, bestAction);

			foreach (var action in actions.Skip(1))
			{
				var value = GetValue(state, action);
				if (value <= bestValue)
					continue;

				bestValue  = value;
				bestAction = action;
			}

			return bestAction;
		}

		public void Save(string path)
		{
			var rows = _q.Select(pair => new QValueRow(pair.Key.State, pair.Key.Action, pair.Value)).ToArray();
			File.WriteAllText(path, JsonSerializer.Serialize(rows));
		}

		public void Load(string path)
		{
			_q.Clear();
			var rows = JsonSerializer.Deserialize<QValueRow[]>(File.ReadAllText(path)) ?? [];
			foreach (var row in rows)
				_q[new StateActionKey(row.State, row.Action)] = row.Value;
		}

		private TAction SelectExploratoryAction(TState state)
		{
			return _random.NextDouble() < Epsilon
				? _environment.ActionSpace.Sample(_random)
				: SelectAction(state);
		}

		private float BestValue(TState state)
		{
			return _environment.ActionSpace.Actions.Max(action => GetValue(state, action));
		}

		private float GetValue(TState state, TAction action)
		{
			return _q.GetValueOrDefault(CreateKey(state, action), 0f);
		}

		private void SetValue(TState state, TAction action, float value)
		{
			_q[CreateKey(state, action)] = value;
		}

		private StateActionKey CreateKey(TState state, TAction action)
		{
			return new StateActionKey(_stateKey(state), action?.ToString() ?? string.Empty);
		}

		private sealed record StateActionKey(string State, string Action);

		private sealed record QValueRow(string State, string Action, float Value);
	}
}
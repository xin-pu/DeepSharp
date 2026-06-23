using System.Diagnostics;
using System.Text.Json;
using DeepSharp.RL.Enumerates;
using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents.Tabular
{
	/// <summary>
	///     Value Iteration demo.
	///     [Deep Reinforcement Learning Hands-On Second Edition] (Russia, Maxim Lapan)
	///     [Chap 5.5]
	/// </summary>
	public class ValueIteration : Agent

	{
		public ValueIteration(Environ<Space, Space> env, int t, float gamma = 0.9f)
			: base(env, "ValueIteration")
		{
			Rewards  = new Dictionary<RewardKey, Reward>();
			Transits = new Dictionary<TransitKey, Dictionary<torch.Tensor, int>>();
			Values   = new Dictionary<torch.Tensor, float>();
			T        = t;
			Gamma    = gamma;
		}

		public int T { get; protected set; }

		public float Gamma { get; protected set; }

		/// <summary>
		///     Reward table.
		/// </summary>
		public Dictionary<RewardKey, Reward> Rewards { get; set; }

		/// <summary>
		///     Transition table.
		/// </summary>
		public Dictionary<TransitKey, Dictionary<torch.Tensor, int>> Transits { get; set; }

		/// <summary>
		///     Value table.
		/// </summary>
		public Dictionary<torch.Tensor, float> Values { get; set; }


		/// <summary>
		///     Select action according to latest observation.
		/// </summary>
		/// <param name="state"></param>
		/// <returns></returns>
		public override Act GetPolicyAct(torch.Tensor state)
		{
			Debug.Assert(Rewards.Count  > 0, "Rewards Table is Empty, You should learn first.");
			Debug.Assert(Transits.Count > 0, "Transits Table is Empty, You should learn first.");
			Debug.Assert(Values.Count   > 0, "Values Table is Empty, You should learn first.");

			// Step 1: Get action space from Transits for current state
			var actionSpace = Transits.Keys
				.Where(a => a.State.Equals(state))
				.ToList();

			var valueDict = actionSpace
				.ToDictionary(a => a.Act, a => GetActionValue(a));
			var maxValue = valueDict.Values.Max();
			var maxActs = valueDict
				.Where(a => Math.Abs(a.Value - maxValue) < 1E-4)
				.Select(a => a.Key)
				.ToList();

			if (maxActs.Count == 1) return new Act(maxActs.First());

			var probs    = Enumerable.Repeat(1f, maxActs.Count).ToArray();
			var actIndex = torch.multinomial(torch.tensor(probs), 1, true).ToInt32();
			return new Act(maxActs[actIndex]);
		}


		public override LearnOutcome Learn()
		{
			var episodes = RunEpisodes(T, PlayMode.Sample);
			UpdateValueIteration();
			return new LearnOutcome(episodes);
		}

		public override void Save(string path)
		{
			var data = new VData
			{
				Rewards = Rewards.Select(kvp => new REntry
				{
					State    = kvp.Key.State.data<float>().ToArray(),
					Act      = kvp.Key.Act.data<float>().ToArray(),
					NewState = kvp.Key.NewState.data<float>().ToArray(),
					Reward   = kvp.Value.Value
				}).ToList(),
				Transits = Transits.Select(kvp => new TEntry
				{
					State = kvp.Key.State.data<float>().ToArray(),
					Act   = kvp.Key.Act.data<float>().ToArray(),
					Targets = kvp.Value.Select(v => new TTarget
					{
						Tensor = v.Key.data<float>().ToArray(),
						Count  = v.Value
					}).ToList()
				}).ToList(),
				Values = Values.Select(kvp => new VEntry
				{
					State = kvp.Key.data<float>().ToArray(),
					Value = kvp.Value
				}).ToList()
			};
			var json = JsonSerializer.Serialize(data);
			File.WriteAllText(path, json);
		}

		public override void Load(string path)
		{
			var json = File.ReadAllText(path);
			var data = JsonSerializer.Deserialize<VData>(json) ?? new VData();
			Rewards = data.Rewards.ToDictionary(
				r => new RewardKey(
					new Observation(torch.tensor(r.State)),
					new Act(torch.tensor(r.Act)),
					new Observation(torch.tensor(r.NewState))),
				r => new Reward(r.Reward));
			Transits = data.Transits.ToDictionary(
				t => new TransitKey(torch.tensor(t.State), torch.tensor(t.Act)),
				t => t.Targets.ToDictionary(v => torch.tensor(v.Tensor), v => v.Count));
			Values = data.Values.ToDictionary(
				v => torch.tensor(v.State), v => v.Value);
		}

		public void Update(Episode episode)
		{
			episode.Steps.ForEach(UpdateTables);
		}


		/// <summary>
		///     Update the value table by iterating over all states.
		/// </summary>
		public void UpdateValueIteration()
		{
			var stateList = Transits.Select(a => a.Key.State).Distinct();

			foreach (var state in stateList)
			{
				var actionList = Transits.Keys
					.Where(a => a.State.Equals(state))
					.Select(a => a.Act);

				var maxStateValue = actionList
					.Select(a => GetActionValue(new TransitKey(state, a)))
					.Max();

				Values[state] = maxStateValue;
			}
		}


		private void UpdateTables(Observation state, Act act, Observation newState, Reward reward)
		{
			var startTensor = state.Value!;
			var newTensor   = newState.Value!;
			var action      = act.Value!;

			// Step 1: Update reward table
			var rewardKey = new RewardKey(state, act, newState);
			var existRewardKey = Rewards.Keys.Where(a =>
					a.Act.Equals(action)        &&
					a.State.Equals(startTensor) &&
					a.NewState.Equals(newTensor))
				.ToList();

			var finalRewardKey = existRewardKey.Any() ? existRewardKey.First() : rewardKey;
			Rewards[finalRewardKey] = reward;


			var transitsKey = new TransitKey(state, act);
			var existTransitKey = Transits.Keys.Where(a =>
					a.Act.Equals(act.Value!) &&
					a.State.Equals(state.Value!))
				.ToList();

			// Step 2: Update transition table
			Dictionary<torch.Tensor, int> sonDict;
			if (existTransitKey.Any())
			{
				sonDict = Transits[existTransitKey.First()];
			}
			else
			{
				sonDict               = new Dictionary<torch.Tensor, int>();
				Transits[transitsKey] = sonDict;
			}

			var newStateKeys = sonDict.Keys
				.Where(a => a.Equals(newState.Value!))
				.ToList();
			if (newStateKeys.Any())
				sonDict[newStateKeys.First()]++;
			else
				sonDict[newTensor] = 1;
		}

		private void UpdateTables(Step step)
		{
			UpdateTables(step.PreState, step.Action, step.PostState, step.Reward);
		}

		/// <summary>
		///     Approximate Q(s,a) = sum of probability * state value for each next state.
		///     By Bellman equation, also equals immediate reward + discounted long-term value.
		/// </summary>
		/// <param name="transitKey"></param>
		/// <returns></returns>
		private float GetActionValue(TransitKey transitKey)
		{
			var targetCounts = getTransit(transitKey);
			var total        = targetCounts.Sum(a => a.Value);
			var activaValue  = 0f;
			foreach (var i in targetCounts)
			{
				var reward = getReward(new RewardKey(transitKey.State, transitKey.Act, i.Key));
				var value  = reward.Value + Gamma * getValue(i.Key);
				activaValue += 1f * i.Value / total * value;
			}

			return activaValue;
		}

		private Dictionary<torch.Tensor, int> getTransit(TransitKey traitKey)
		{
			try
			{
				var key = Transits.Keys
					.First(a => a.Act.Equals(traitKey.Act) &&
					            a.State.Equals(traitKey.State));

				return Transits[key];
			}
			catch (Exception)
			{
				return new Dictionary<torch.Tensor, int> { [traitKey.State] = 0 };
			}
		}

		private Reward getReward(RewardKey rewardKey)
		{
			try
			{
				var key = Rewards.Keys
					.First(a => a.Act.Equals(rewardKey.Act)     &&
					            a.State.Equals(rewardKey.State) &&
					            a.NewState.Equals(rewardKey.NewState));
				return Rewards[key];
			}
			catch (Exception)
			{
				return new Reward(0);
			}
		}

		private float getValue(torch.Tensor observation)
		{
			try
			{
				var key = Values.Keys.First(a => a.Equals(observation));
				return Values[key];
			}
			catch
			{
				return 0;
			}
		}
	}

	internal record VData
	{
		public List<REntry> Rewards { get; init; } = new();

		public List<TEntry> Transits { get; init; } = new();

		public List<VEntry> Values { get; init; } = new();
	}

	internal record REntry
	{
		public float[] State { get; init; } = Array.Empty<float>();

		public float[] Act { get; init; } = Array.Empty<float>();

		public float[] NewState { get; init; } = Array.Empty<float>();

		public float Reward { get; init; }
	}

	internal record TEntry
	{
		public float[] State { get; init; } = Array.Empty<float>();

		public float[] Act { get; init; } = Array.Empty<float>();

		public List<TTarget> Targets { get; init; } = new();
	}

	internal record TTarget
	{
		public float[] Tensor { get; init; } = Array.Empty<float>();

		public int Count { get; init; }
	}

	internal record VEntry
	{
		public float[] State { get; init; } = Array.Empty<float>();

		public float Value { get; init; }
	}
}
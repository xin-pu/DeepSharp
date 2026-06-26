using System.Text.Json;
using RLSharp.Torch.Environs;

namespace RLSharp.Torch.Agents.Tabular
{
	/// <summary>
	///     Tabular agent base class â€?uses a Q-table for state-action value lookup.
	/// </summary>
	public abstract class TabularAgent : Agent
	{
		protected TabularAgent(EnvironmentBase<Space, Space> env, string name)
			: base(env, name)
		{
			QTable = new QTable();
		}

		public QTable QTable { get; protected set; }

		/// <summary>
		///     argmax_{a'} Q(state, a') â€?returns the best action or random if Q is zero.
		/// </summary>
		public override ActionValue GetPolicyAct(torch.Tensor state)
		{
			var action = QTable.GetBestAct(state);
			return action ?? GetSampleAct();
		}

		public override void Save(string path)
		{
			var data = QTable.Return.Select(kvp => new QEntry
			{
				State       = kvp.Key.State.data<float>().ToArray(),
				ActionValue = kvp.Key.ActionValue.data<float>().ToArray(),
				Value       = kvp.Value
			}).ToList();
			var json = JsonSerializer.Serialize(data);
			File.WriteAllText(path, json);
		}

		public override void Load(string path)
		{
			var json = File.ReadAllText(path);
			var data = JsonSerializer.Deserialize<List<QEntry>>(json) ?? new List<QEntry>();
			QTable = new QTable();
			foreach (var entry in data)
			{
				var key = new TransitKey(torch.tensor(entry.State), torch.tensor(entry.ActionValue));
				QTable[key] = entry.Value;
			}
		}

		private record QEntry
		{
			public float[] State { get; init; } = Array.Empty<float>();

			public float[] ActionValue { get; init; } = Array.Empty<float>();

			public float Value { get; init; }
		}
	}
}
using System.Text;
using DeepSharp.RL.Environs.Spaces;

namespace DeepSharp.RL.Environs
{
	/// <summary>
	///     K-armed bandit: each arm has a random probability in [0, 1] of yielding reward.
	/// </summary>
	public sealed class KArmedBandit : Environ<Space, Space>
	{
		public KArmedBandit(int k, DeviceType deviceType = DeviceType.CPU)
			: base("KArmedBandit", deviceType)
		{
			bandits          = new Bandit[k];
			ActionSpace      = new Discrete(k, deviceType: deviceType);
			ObservationSpace = new Box(0, 1, new long[] { k }, deviceType);
			Create(k);
			Reset();
		}


		public KArmedBandit(double[] probs, DeviceType deviceType = DeviceType.CPU)
			: base("KArmedBandit", deviceType)
		{
			var k = probs.Length;
			bandits          = new Bandit[k];
			ActionSpace      = new Discrete(k, deviceType: deviceType);
			ObservationSpace = new Box(0, 1, new long[] { k }, deviceType);
			Create(probs);
			Reset();
		}

		private Bandit[] bandits { get; }

		public Bandit this[int k] => bandits[k];


		private void Create(int k)
		{
			var random = new Random();
			foreach (var i in Enumerable.Range(0, k))
				bandits[i] = new Bandit($"{i}", random.NextDouble());
		}

		private void Create(double[] probs)
		{
			foreach (var i in Enumerable.Range(0, probs.Length))
				bandits[i] = new Bandit($"{i}", probs[i]);
		}


		/// <summary>
		///     Sum the observation tensor as the reward (number of coins from all bandits).
		/// </summary>
		/// <param name="observation"></param>
		/// <returns></returns>
		public override Reward GetReward(Observation observation)
		{
			var sum = observation.Value!.to_type(torch.ScalarType.Float32)
				.sum()
				.item<float>();
			var reward = new Reward(sum);
			return reward;
		}

		/// <summary>
		///     Average reward across all steps in the episode (for evaluation).
		/// </summary>
		/// <param name="episode"></param>
		/// <returns></returns>
		public override float GetReturn(Episode episode)
		{
			return episode.Steps.Average(a => a.Reward.Value);
		}

		/// <summary>
		/// </summary>
		/// <param name="act">Action containing the selected bandit arm index.</param>
		/// <returns>Observation with 0 or 1 for the selected arm.</returns>
		public override Observation Update(Act act)
		{
			var obs    = new float[ObservationSpace!.N];
			var index  = act.Value!.ToInt64();
			var bandit = bandits[index];
			var value  = bandit.Step();
			obs[index] = value;

			var obsTensor = torch.from_array(obs, torch.ScalarType.Float32).to(Device);
			return new Observation(obsTensor);
		}


		/// <summary>
		///     Complete after 20 steps.
		/// </summary>
		/// <param name="epoch"></param>
		/// <returns></returns>
		public override bool IsComplete(int epoch)
		{
			return epoch >= 20;
		}


		public override string ToString()
		{
			var str = new StringBuilder();
			str.AppendLine(base.ToString());
			str.Append(string.Join("\r\n", bandits.Select(a => $"\t{a}")));
			return str.ToString();
		}
	}
}
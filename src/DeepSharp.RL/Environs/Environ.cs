using System.Text;

namespace DeepSharp.RL.Environs
{
	/// <summary>
	///     Base environment providing observations and rewards.
	/// </summary>
	public abstract class Environ<T1, T2>
		where T1 : Space
		where T2 : Space
	{
		public Action<Step>? CallBack;

		protected Environ(string name, DeviceType deviceType = DeviceType.CUDA)
		{
			Name            = name;
			Device          = new torch.Device(deviceType);
			Reward          = new Reward(0);
			ObservationList = new List<Observation>();
		}

		protected Environ(string name)
			: this(name, DeviceType.CPU)
		{
		}


		public string Name { get; set; }

		public torch.Device Device { get; set; }

		public T1? ActionSpace { get; protected set; }

		public T2? ObservationSpace { get; protected set; }


		/// <summary>
		///     Current observation.
		/// </summary>
		public Observation? Observation { get; set; }

		/// <summary>
		///     Current reward.
		/// </summary>
		public Reward Reward { get; set; }

		/// <summary>
		///     History of all observations in the current episode.
		/// </summary>
		public List<Observation> ObservationList { get; set; }

		public int Life => ObservationList.Count;


		/// <summary>
		///     Reset environment to initial state and return the initial observation.
		/// </summary>
		public virtual Observation Reset()
		{
			Observation     = new Observation(ObservationSpace!.Generate());
			ObservationList = new List<Observation> { Observation };
			Reward          = new Reward(0);
			return Observation;
		}

		/// <summary>
		///     Compute the total return for an episode.
		/// </summary>
		/// <param name="episode"></param>
		/// <returns></returns>
		public abstract float GetReturn(Episode episode);

		public virtual Act SampleAct()
		{
			return new Act(ActionSpace!.Sample());
		}

		/// <summary>
		///     Execute one step: agent provides an action, environment returns next state and reward.
		/// </summary>
		/// <param name="act">Action from the agent.</param>
		/// <returns>The resulting step.</returns>
		public virtual Step Step(Act act, int epoch)
		{
			var state    = Observation!;
			var stateNew = Update(act);
			var reward   = GetReward(stateNew);
			var complete = IsComplete(epoch);
			var step     = new Step(state, act, stateNew, reward, complete);
			ObservationList.Add(stateNew);
			Observation = stateNew;
			return step;
		}


		/// <summary>
		///     Update environment observation according to one action from agent.
		/// </summary>
		/// <param name="act">Action from the policy.</param>
		/// <returns>New observation after the action.</returns>
		public abstract Observation Update(Act act);


		/// <summary>
		///     Calculate the one-step reward from an observation.
		/// </summary>
		/// <param name="observation">One observation.</param>
		/// <returns>One reward.</returns>
		public abstract Reward GetReward(Observation observation);


		/// <summary>
		///     Check whether the episode should end.
		/// </summary>
		/// <param name="epoch">Current time step.</param>
		/// <returns>True if the episode is complete.</returns>
		public abstract bool IsComplete(int epoch);


		public override string ToString()
		{
			var str = new StringBuilder();
			str.AppendLine($"{Name}\tLife:{Life}");
			str.AppendLine(new string('-', 30));
			str.Append($"State:\t{Observation!.Value!.ToString(torch.numpy)}");
			return str.ToString();
		}
	}
}
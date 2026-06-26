using System.Text;

namespace RLSharp.Torch.Environs
{
	/// <summary>
	///     Base environment providing observations and rewards.
	/// </summary>
	public abstract class EnvironmentBase<T1, T2>
		where T1 : Space
		where T2 : Space
	{
		public Action<Step>? CallBack;

		protected EnvironmentBase(string name, DeviceType deviceType = DeviceType.CPU)
		{
			Name            = name;
			Device          = new torch.Device(deviceType);
			Reward          = new Reward(0);
			ObservationList = new List<ObservationValue>();
		}

		protected EnvironmentBase(string name)
			: this(name, DeviceType.CPU)
		{
		}


		public string Name { get; set; }

		public torch.Device Device { get; set; }

		public T1? ActionSpace { get; protected set; }

		public T2? ObservationSpace { get; protected set; }


		/// <summary>
		///     Current ObservationValue.
		/// </summary>
		public ObservationValue? ObservationValue { get; set; }

		/// <summary>
		///     Current reward.
		/// </summary>
		public Reward Reward { get; set; }

		/// <summary>
		///     History of all observations in the current episode.
		/// </summary>
		public List<ObservationValue> ObservationList { get; set; }

		public int Life => ObservationList.Count;


		/// <summary>
		///     Reset environment to initial state and return the initial ObservationValue.
		/// </summary>
		public virtual ObservationValue Reset()
		{
			ObservationValue = new ObservationValue(ObservationSpace!.Generate());
			ObservationList  = new List<ObservationValue> { ObservationValue };
			Reward           = new Reward(0);
			return ObservationValue;
		}

		/// <summary>
		///     Compute the total return for an episode.
		/// </summary>
		/// <param name="episode"></param>
		/// <returns></returns>
		public abstract float GetReturn(Episode episode);

		public virtual ActionValue SampleAct()
		{
			return new ActionValue(ActionSpace!.Sample());
		}

		/// <summary>
		///     Execute one step: agent provides an action, environment returns next state and reward.
		/// </summary>
		/// <param name="ActionValue">Action from the agent.</param>
		/// <returns>The resulting step.</returns>
		public virtual Step Step(ActionValue ActionValue, int epoch)
		{
			var state    = ObservationValue!;
			var stateNew = Update(ActionValue);
			var reward   = GetReward(stateNew);
			var complete = IsComplete(epoch);
			var step     = new Step(state, ActionValue, stateNew, reward, complete);
			ObservationList.Add(stateNew);
			ObservationValue = stateNew;
			return step;
		}


		/// <summary>
		///     Update environment ObservationValue according to one action from agent.
		/// </summary>
		/// <param name="ActionValue">Action from the policy.</param>
		/// <returns>New ObservationValue after the action.</returns>
		public abstract ObservationValue Update(ActionValue ActionValue);


		/// <summary>
		///     Calculate the one-step reward from an ObservationValue.
		/// </summary>
		/// <param name="ObservationValue">One ObservationValue.</param>
		/// <returns>One reward.</returns>
		public abstract Reward GetReward(ObservationValue ObservationValue);


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
			str.Append($"State:\t{ObservationValue!.Value!.ToString(torch.numpy)}");
			return str.ToString();
		}
	}
}
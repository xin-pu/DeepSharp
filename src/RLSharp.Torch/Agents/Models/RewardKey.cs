using RLSharp.Torch.Environs;

namespace RLSharp.Torch.Agents
{
	/// <summary>
	///     Reward table composite key.
	/// </summary>
	public struct RewardKey
	{
		public RewardKey(torch.Tensor state, torch.Tensor action, torch.Tensor newState)
		{
			State    = state;
			ActionValue = action;
			NewState = newState;
		}

		public RewardKey(ObservationValue state, ActionValue action, ObservationValue newState)
		{
			State    = state.Value!;
			ActionValue = action.Value!;
			NewState = newState.Value!;
		}

		public torch.Tensor State { get; set; }

		public torch.Tensor ActionValue { get; set; }

		public torch.Tensor NewState { get; set; }


		public override string ToString()
		{
			var state    = State.ToString(torch.numpy);
			var action   = ActionValue.ToString(torch.numpy);
			var newState = NewState.ToString(torch.numpy);

			return $"{state} \r\n {action}  \r\n {newState}";
		}

		public static bool operator ==(RewardKey x, RewardKey y)
		{
			return x.Equals(y);
		}

		public static bool operator !=(RewardKey x, RewardKey y)
		{
			return !x.Equals(y);
		}

		public bool Equals(RewardKey other)
		{
			return State.Equals(other.State) && ActionValue.Equals(other.ActionValue) && NewState.Equals(other.NewState);
		}


		public override bool Equals(object? obj)
		{
			if (obj is RewardKey input)
				return Equals(input);
			return false;
		}

		public override int GetHashCode()
		{
			return -1;
		}
	}
}
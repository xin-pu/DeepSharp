using RLSharp.Torch.Environs;

namespace RLSharp.Torch.Agents
{
	/// <summary>
	///     Transition key — composite key of state + action for table lookups.
	/// </summary>
	public class TransitKey
	{
		public TransitKey(ObservationValue state, ActionValue action)
		{
			State = state.Value!;
			ActionValue = action.Value!;
		}

		public TransitKey(torch.Tensor state, torch.Tensor action)
		{
			State = state;
			ActionValue = action;
		}

		public torch.Tensor State { get; protected set; }

		public torch.Tensor ActionValue { get; protected set; }


		public static bool operator ==(TransitKey x, TransitKey y)
		{
			return x.Equals(y);
		}

		public static bool operator !=(TransitKey x, TransitKey y)
		{
			return !x.Equals(y);
		}

		public bool Equals(TransitKey other)
		{
			return State.Equals(other.State) && ActionValue.Equals(other.ActionValue);
		}

		public override bool Equals(object? obj)
		{
			if (obj is TransitKey input)
				return Equals(input);
			return false;
		}

		public override int GetHashCode()
		{
			return -1;
		}

		public override string ToString()
		{
			return $"{State.ToString(torch.numpy)},{ActionValue.ToString(torch.numpy)}";
		}
	}
}
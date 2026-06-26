namespace RLSharp.Torch.Encoding
{
	public sealed class DelegateStateEncoder<TState> : IStateEncoder<TState>
	{
		private readonly Func<TState, float[]> _encode;

		public DelegateStateEncoder(long inputSize, Func<TState, float[]> encode)
		{
			if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
			InputSize = inputSize;
			_encode   = encode ?? throw new ArgumentNullException(nameof(encode));
		}

		public long InputSize { get; }

		public torch.Tensor Encode(TState state)
		{
			var values = _encode(state);
			if (values.Length != InputSize)
				throw new InvalidOperationException(
					$"Encoded state length {values.Length} does not match expected input size {InputSize}.");

			return torch.tensor(values, torch.ScalarType.Float32);
		}
	}
}
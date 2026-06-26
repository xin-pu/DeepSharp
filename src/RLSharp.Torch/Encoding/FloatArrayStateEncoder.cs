namespace RLSharp.Torch.Encoding
{
	public sealed class FloatArrayStateEncoder : IStateEncoder<float[]>
	{
		public FloatArrayStateEncoder(long inputSize)
		{
			if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
			InputSize = inputSize;
		}

		public long InputSize { get; }

		public torch.Tensor Encode(float[] state)
		{
			ArgumentNullException.ThrowIfNull(state);
			if (state.Length != InputSize)
				throw new InvalidOperationException(
					$"State length {state.Length} does not match expected input size {InputSize}.");

			return torch.tensor(state, torch.ScalarType.Float32);
		}
	}
}
namespace RLSharp.Torch.Encoding
{
	public interface IStateEncoder<TState>
	{
		long InputSize { get; }

		torch.Tensor Encode(TState state);
	}
}
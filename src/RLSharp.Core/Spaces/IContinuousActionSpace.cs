namespace RLSharp.Core.Spaces
{
	public interface IContinuousActionSpace<TAction> : IActionSpace<TAction>
	{
		int Dimensions { get; }

		TAction FromVector(IReadOnlyList<float> values);

		IReadOnlyList<float> ToVector(TAction action);
	}
}
namespace RLSharp.Core.Spaces
{
	public interface IActionSpace<TAction>
	{
		IReadOnlyList<TAction> Actions { get; }

		TAction Sample(Random random);
	}
}
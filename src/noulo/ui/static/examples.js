// Shared, side-effect-free examples: the UI and regression tests use the same data.
export const DEFAULT_RUBRIC = ["insignificant", "low", "medium", "high", "critical"];

export const EXAMPLES = {
  noul: () => ({
    input: "I checked my account and you have taken the subscription payment twice.",
    proposition: "The customer reports being charged more than once.",
  }),
  choice: () => ({
    input: "The customer says their subscription payment was taken twice.",
    question: "Which department should handle this?",
    options: [
      { id: "A", text: "Resolving an existing charge, payment, invoice or refund" },
      { id: "B", text: "Fixing a software crash, error or malfunction" },
      { id: "C", text: "Getting a quote or purchasing new products or additional licences" },
    ],
  }),
  score: () => ({
    input: "The production system is unavailable for every customer.",
    question: "How severe is this incident?",
    rubric: [...DEFAULT_RUBRIC],
  }),
};

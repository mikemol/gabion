---
title: "TTL-Derived Kernel Report"
subtitle: "Direct mechanical derivation from ontology, shapes, and example TTL graphs"
date: "2026-03-10T18:55:42"
generator: "scripts/ttl_to_pandoc.py"
---

## Input Files
| File | Resource blocks | Stage markers |
| --- | --- | --- |
| lg_kernel_ontology_cut_elim-1.ttl | 514 | 11 |
| lg_kernel_shapes_cut_elim-1.ttl | 159 | 11 |
| lg_kernel_example_cut_elim-1.ttl | 299 | 11 |

## Ontology Inventory
| Metric | Value |
| --- | --- |
| Classes (rdfs:Class) | 204 |
| Properties (rdf:Property) | 309 |
| Resources total | 514 |
| Namespace cat | 11 |
| Namespace lg | 502 |
| Namespace sh | 1 |

## Node Shape Inventory
| NodeShape | Target class | Constrained class | Properties | SPARQL rules |
| --- | --- | --- | --- | --- |
| cat:FunctorShape | cat:Functor | cat:Functor | 2 | 0 |
| cat:MorphismShape | cat:Morphism | cat:Morphism | 2 | 0 |
| lg:AbsenceEvaluationStepShape | lg:AbsenceEvaluationStep | lg:AbsenceEvaluationStep | 2 | 0 |
| lg:AbsenceEvaluationTraceShape | lg:AbsenceEvaluationTrace | lg:AbsenceEvaluationTrace | 2 | 0 |
| lg:AbsorptionObligationShape | lg:AbsorptionObligation | lg:AbsorptionObligation | 3 | 0 |
| lg:AdjointPairShape | lg:AdjointPair | lg:AdjointPair | 3 | 0 |
| lg:AntiJoinPatternShape | lg:AntiJoinPattern | lg:AntiJoinPattern | 2 | 0 |
| lg:AugmentedRulePolarityShape | lg:AugmentedRule | lg:AugmentedRule | 1 | 0 |
| lg:AugmentedRuleShape | lg:AugmentedRule | lg:AugmentedRule | 4 | 2 |
| lg:BindingWitnessShape | lg:BindingWitness | lg:BindingWitness | 3 | 0 |
| lg:BoundednessObligationShape | lg:BoundednessObligation | lg:BoundednessObligation | 3 | 0 |
| lg:CategoricalRuleAtlasShape | lg:CategoricalRule | lg:CategoricalRule | 1 | 0 |
| lg:CategoricalRuleShape | lg:CategoricalRule | lg:CategoricalRule | 2 | 0 |
| lg:ClosedRuleCellQuotientShape | lg:ClosedRuleCell | lg:ClosedRuleCell | 6 | 7 |
| lg:ClosedRuleCellShape | lg:ClosedRuleCell | lg:ClosedRuleCell | 2 | 0 |
| lg:ConceptLatticeShape | lg:ConceptLattice | lg:ConceptLattice | 7 | 0 |
| lg:ConceptLatticeTheoremShape | lg:ConceptLattice | lg:ConceptLattice | 9 | 9 |
| lg:ConservativeProjectionObligationShape | lg:ConservativeProjectionObligation | lg:ConservativeProjectionObligation | 6 | 0 |
| lg:ConstraintDenotationShape | lg:ConstraintDenotation | lg:ConstraintDenotation | 2 | 1 |
| lg:ConstructiveCarryingObligationShape | lg:ConstructiveCarryingObligation | lg:ConstructiveCarryingObligation | 5 | 0 |
| lg:CutAdmissibilityObligationShape | lg:CutAdmissibilityObligation | lg:CutAdmissibilityObligation | 3 | 0 |
| lg:CutApplicationShape | lg:CutApplication | lg:CutApplication | 8 | 3 |
| lg:CutEliminationObligationShape | lg:CutEliminationObligation | lg:CutEliminationObligation | 3 | 0 |
| lg:CutEliminationWitnessShape | lg:CutEliminationWitness | lg:CutEliminationWitness | 4 | 1 |
| lg:DenotationalQuotientShape | lg:DenotationalQuotient | lg:DenotationalQuotient | 4 | 0 |
| lg:EmptyQueryResultSetShape | lg:EmptyQueryResultSet | lg:EmptyQueryResultSet | 2 | 0 |
| lg:EmptyResultConstructivityObligationShape | lg:EmptyResultConstructivityObligation | lg:EmptyResultConstructivityObligation | 8 | 0 |
| lg:EmptyResultProofQueryShape | lg:EmptyResultProofQuery | lg:EmptyResultProofQuery | 5 | 1 |
| lg:EmptyResultWitnessShape | lg:EmptyResultWitness | lg:EmptyResultWitness | 4 | 1 |
| lg:EmptyWedgeProductShape | lg:EmptyWedgeProduct | lg:EmptyWedgeProduct | 1 | 0 |
| lg:EvolutionStepShape | lg:EvolutionStep | lg:EvolutionStep | 3 | 2 |
| lg:ExistentialImageIntroTermShape | lg:ExistentialImageIntroTerm | lg:ExistentialImageIntroTerm | 6 | 1 |
| lg:ExistentialImageShape | lg:ExistentialImage | lg:ExistentialImage | 2 | 0 |
| lg:ExistsWitnessedQueryShape | lg:ExistsWitnessedQuery | lg:ExistsWitnessedQuery | 7 | 2 |
| lg:FormalConceptShape | lg:FormalConcept | lg:FormalConcept | 3 | 0 |
| lg:FunctorialSubproofReuseObligationShape | lg:FunctorialSubproofReuseObligation | lg:FunctorialSubproofReuseObligation | 1 | 0 |
| lg:HistoricalAssertionShape | lg:HistoricalAssertion | lg:HistoricalAssertion | 3 | 0 |
| lg:HistoryBoundaryCompletenessObligationShape | lg:HistoryBoundaryCompletenessObligation | lg:HistoryBoundaryCompletenessObligation | 3 | 0 |
| lg:HistoryExtensionShape | lg:HistoryExtension | lg:HistoryExtension | 2 | 0 |
| lg:HistoryImmutabilityObligationShape | lg:HistoryImmutabilityObligation | lg:HistoryImmutabilityObligation | 1 | 0 |
| lg:HistoryLineageShape | lg:HistoryLineage | lg:HistoryLineage | 2 | 1 |
| lg:HistoryStampIntroTermShape | lg:HistoryStampIntroTerm | lg:HistoryStampIntroTerm | 3 | 0 |
| lg:HistoryStampShape | lg:HistoryStamp | lg:HistoryStamp | 4 | 2 |
| lg:HistoryStateShape | lg:HistoryState | lg:HistoryState | 3 | 0 |
| lg:InternalDerivationRealizationShape | lg:PolicyIndexedDerivation | lg:PolicyIndexedDerivation | 1 | 1 |
| lg:InternalDerivationShape | lg:InternalDerivation | lg:InternalDerivation | 1 | 0 |
| lg:InternalProofCalculusShape | lg:InternalProofCalculus | lg:InternalProofCalculus | 2 | 0 |
| lg:InternalProofTermShape | lg:InternalProofTerm | lg:InternalProofTerm | 1 | 0 |
| lg:JoinAssociativityObligationShape | lg:JoinAssociativityObligation | lg:JoinAssociativityObligation | 2 | 0 |
| lg:JoinCommutativityObligationShape | lg:JoinCommutativityObligation | lg:JoinCommutativityObligation | 2 | 0 |
| lg:JoinIdempotenceObligationShape | lg:JoinIdempotenceObligation | lg:JoinIdempotenceObligation | 2 | 0 |
| lg:JoinPatternShape | lg:JoinPattern | lg:JoinPattern | 2 | 0 |
| lg:LoweringFunctorShape | lg:LoweringFunctor | lg:LoweringFunctor | 2 | 0 |
| lg:LoweringSoundnessObligationShape | lg:LoweringSoundnessObligation | lg:LoweringSoundnessObligation | 6 | 0 |
| lg:MeetAssociativityObligationShape | lg:MeetAssociativityObligation | lg:MeetAssociativityObligation | 2 | 0 |
| lg:MeetCommutativityObligationShape | lg:MeetCommutativityObligation | lg:MeetCommutativityObligation | 2 | 0 |
| lg:MeetIdempotenceObligationShape | lg:MeetIdempotenceObligation | lg:MeetIdempotenceObligation | 2 | 0 |
| lg:NegationIntroTermShape | lg:NegationIntroTerm | lg:NegationIntroTerm | 5 | 1 |
| lg:NegationPolicySemanticsObligationShape | lg:NegationPolicySemanticsObligation | lg:NegationPolicySemanticsObligation | 4 | 0 |
| lg:NegationPolicyShape | lg:NegationPolicy | lg:NegationPolicy | 3 | 0 |
| lg:NormalizationWitnessShape | lg:NormalizationWitness | lg:NormalizationWitness | 2 | 0 |
| lg:NotExistsSemanticPolicyShape | lg:NotExistsWitnessedQuery | lg:NotExistsWitnessedQuery | 3 | 3 |
| lg:NotExistsWitnessedQueryShape | lg:NotExistsWitnessedQuery | lg:NotExistsWitnessedQuery | 9 | 3 |
| lg:OrderCompatibilityObligationShape | lg:OrderCompatibilityObligation | lg:OrderCompatibilityObligation | 4 | 0 |
| lg:PolicyCertificateShape | lg:PolicyCertificate | lg:PolicyCertificate | 6 | 0 |
| lg:PolicyIndexedConstructiveDerivationObligationShape | lg:PolicyIndexedConstructiveDerivationObligation | lg:PolicyIndexedConstructiveDerivationObligation | 8 | 0 |
| lg:PolicyIndexedDerivationShape | lg:PolicyIndexedDerivation | lg:PolicyIndexedDerivation | 6 | 2 |
| lg:PolicyIndexedProofCalculusShape | lg:PolicyIndexedProofCalculus | lg:PolicyIndexedProofCalculus | 2 | 0 |
| lg:PolicyIndexedProofCarryingQueryRealizationShape | lg:PolicyIndexedProofCarryingQuery | lg:PolicyIndexedProofCarryingQuery | 1 | 1 |
| lg:PolicyIndexedProofCarryingQueryShape | lg:PolicyIndexedProofCarryingQuery | lg:PolicyIndexedProofCarryingQuery | 7 | 5 |
| lg:PolicyIndexedProofTermShape | lg:PolicyIndexedProofTerm | lg:PolicyIndexedProofTerm | 2 | 0 |
| lg:PolicyPreservingLoweringObligationShape | lg:PolicyPreservingLoweringObligation | lg:PolicyPreservingLoweringObligation | 9 | 0 |
| lg:PolicyReflectiveCompatibilityObligationShape | lg:PolicyReflectiveCompatibilityObligation | lg:PolicyReflectiveCompatibilityObligation | 10 | 0 |
| lg:PolicyScopedJudgmentShape | lg:PolicyScopedJudgment | lg:PolicyScopedJudgment | 3 | 0 |
| lg:ProofAssumptionShape | lg:ProofAssumption | lg:ProofAssumption | 3 | 1 |
| lg:ProofCarryingQueryHistoryBoundaryShape | lg:ProofCarryingQuery | lg:ProofCarryingQuery | 2 | 2 |
| lg:ProofCarryingQueryShape | lg:ProofCarryingQuery | lg:ProofCarryingQuery | 8 | 3 |
| lg:ProofConstructorShape | lg:ProofConstructor | lg:ProofConstructor | 0 | 0 |
| lg:ProofContextShape | lg:ProofContext | lg:ProofContext | 2 | 0 |
| lg:ProofObligationShape | lg:ProofObligation | lg:ProofObligation | 1 | 0 |
| lg:ProofTermAdequacyObligationShape | lg:ProofTermAdequacyObligation | lg:ProofTermAdequacyObligation | 3 | 0 |
| lg:ProofTermExpressionShape | lg:ProofTermExpression | lg:ProofTermExpression | 2 | 0 |
| lg:ProofVariableShape | lg:ProofVariable | lg:ProofVariable | 1 | 0 |
| lg:QueryEvaluationTraceShape | lg:QueryEvaluationTrace | lg:QueryEvaluationTrace | 3 | 1 |
| lg:QueryRenderingShape | lg:QueryRendering | lg:QueryRendering | 2 | 0 |
| lg:QueryResultSetCardinalityShape | lg:QueryResultSet | lg:QueryResultSet | 1 | 0 |
| lg:QueryResultSetShape | lg:QueryResultSet | lg:QueryResultSet | 1 | 0 |
| lg:QueryVariableShape | lg:QueryVariable | lg:QueryVariable | 1 | 0 |
| lg:QueryWedgeProductShape | lg:QueryWedgeProduct | lg:QueryWedgeProduct | 3 | 1 |
| lg:QuotientProjectionShape | lg:QuotientProjection | lg:QuotientProjection | 1 | 0 |
| lg:QuotientProjectionTheoremShape | lg:QuotientProjection | lg:QuotientProjection | 1 | 1 |
| lg:QuotientUniversalityObligationShape | lg:QuotientUniversalityObligation | lg:QuotientUniversalityObligation | 5 | 0 |
| lg:ReflectionFunctorShape | lg:ReflectionFunctor | lg:ReflectionFunctor | 2 | 0 |
| lg:ReflectionShape | lg:CoreTerm | lg:CoreTerm | 1 | 0 |
| lg:ReflectiveQuotientObligationShape | lg:ReflectiveQuotientObligation | lg:ReflectiveQuotientObligation | 7 | 0 |
| lg:ReflectiveSHACLBoundaryShape | lg:ReflectiveSHACLBoundary | lg:ReflectiveSHACLBoundary | 2 | 2 |
| lg:RulePolarityLatticeShape | lg:RulePolarity | lg:RulePolarity | 1 | 1 |
| lg:RulePolarityShape | lg:RulePolarity | lg:RulePolarity | 9 | 4 |
| lg:SHACLBoundaryShape | lg:SHACLBoundary | lg:SHACLBoundary | 8 | 5 |
| lg:SHACLSurjectivityObligationShape | lg:SHACLSurjectivityObligation | lg:SHACLSurjectivityObligation | 6 | 0 |
| lg:SPARQLAlgebraExpressionShape | lg:SPARQLAlgebraExpression | lg:SPARQLAlgebraExpression | 0 | 0 |
| lg:SPARQLConstraintShape | lg:SPARQLConstraint | lg:SPARQLConstraint | 3 | 2 |
| lg:SelectQueryShape | lg:SelectQuery | lg:SelectQuery | 2 | 0 |
| lg:SemanticAtlasShape | lg:SemanticAtlas | lg:SemanticAtlas | 3 | 0 |
| lg:SemanticPolicyBundleShape | lg:SemanticPolicyBundle | lg:SemanticPolicyBundle | 3 | 3 |
| lg:SemanticPolicyCoherenceObligationShape | lg:SemanticPolicyCoherenceObligation | lg:SemanticPolicyCoherenceObligation | 5 | 0 |
| lg:SequentialProofCompositionShape | lg:SequentialProofComposition | lg:SequentialProofComposition | 4 | 2 |
| lg:SharedSubproofReuseObligationShape | lg:SharedSubproofReuseObligation | lg:SharedSubproofReuseObligation | 2 | 0 |
| lg:SharedSubproofShape | lg:SharedSubproof | lg:SharedSubproof | 2 | 0 |
| lg:SolutionMappingShape | lg:SolutionMapping | lg:SolutionMapping | 1 | 0 |
| lg:SubproofReuseCategoryShape | lg:SubproofReuseCategory | lg:SubproofReuseCategory | 2 | 0 |
| lg:SubproofReuseMorphismShape | lg:SubproofReuseMorphism | lg:SubproofReuseMorphism | 4 | 1 |
| lg:SubstitutionApplicationShape | lg:SubstitutionApplication | lg:SubstitutionApplication | 7 | 3 |
| lg:SubstitutionCompatibilityObligationShape | lg:SubstitutionCompatibilityObligation | lg:SubstitutionCompatibilityObligation | 3 | 0 |
| lg:SubstitutionSoundnessObligationShape | lg:SubstitutionSoundnessObligation | lg:SubstitutionSoundnessObligation | 4 | 0 |
| lg:SubstitutionWitnessShape | lg:SubstitutionWitness | lg:SubstitutionWitness | 3 | 0 |
| lg:SupportReflectionIntroTermShape | lg:SupportReflectionIntroTerm | lg:SupportReflectionIntroTerm | 5 | 1 |
| lg:SupportReflectionPolicyShape | lg:SupportReflectionPolicy | lg:SupportReflectionPolicy | 2 | 0 |
| lg:TriplePatternShape | lg:TriplePattern | lg:TriplePattern | 3 | 0 |
| lg:TypingJudgmentShape | lg:TypingJudgment | lg:TypingJudgment | 4 | 0 |
| lg:TypingRuleShape | lg:TypingRule | lg:TypingRule | 1 | 0 |

## Example Stages

### Prelude
Resource blocks in stage: 24.
| Subject | rdf:type | Key outgoing links |
| --- | --- | --- |
| lg:W | lg:WorldCategory | - |
| lg:W0 | lg:WorldState | cat:inCategory -> lg:W |
| lg:W1 | lg:WorldState | cat:inCategory -> lg:W |
| lg:eatTransition | lg:ExecutableTransition | cat:cod -> lg:W1 ; cat:dom -> lg:W0 ; cat:inCategory -> lg:W |
| lg:G0 | lg:GrammarState | - |
| lg:G1 | lg:GrammarState | - |
| lg:eatIdentifier | lg:Identifier | - |
| lg:eatSyntaxClause | lg:SyntacticRule | - |
| lg:eatTypingClause | lg:TypingRule | lg:fiberOverSyntaxClause -> lg:eatSyntaxClause |
| lg:eatCategoricalClause | lg:CategoricalRule | lg:denotesTransition -> lg:eatTransition ; lg:indexedByTypingClause -> lg:eatTypingClause |
| lg:EatRule | lg:AugmentedRule | lg:hasCategoricalClause -> lg:eatCategoricalClause ; lg:hasSyntaxClause -> lg:eatSyntaxClause ; lg:hasTypingClause -> lg:eatTypingClause |
| lg:G0 | - | lg:hasRule -> lg:EatRule |
| lg:G1 | - | lg:hasEmergentIdentifier -> lg:eatIdentifier ; lg:hasRule -> lg:EatRule |
| lg:eatEvolution | lg:EvolutionStep | cat:cod -> lg:G1 ; cat:dom -> lg:G0 ; lg:fromGrammarState -> lg:G0 |
| lg:eatTerm | lg:CoreTerm | lg:reifiesAsHandle -> lg:eatHandle |
| lg:eatHandle | lg:Handle | lg:reflectsHandle -> lg:eatTerm |
| lg:eatVerbType | lg:SemanticType | - |
| lg:eatTypingJudgment | lg:TypingJudgment | lg:judgesInWorldState -> lg:W0 ; lg:judgesTerm -> lg:eatTerm ; lg:judgesType -> lg:eatVerbType |
| lg:thisVar | lg:ReservedVariable | lg:variableName -> "this" |
| lg:demoRuleVar | lg:QueryVariable | lg:variableName -> "r" |
| lg:demoGrammarVar | lg:QueryVariable | lg:variableName -> "g2" |
| lg:demoRulePersistenceQuery | lg:SelectQuery | lg:projectsVariable -> lg:thisVar ; lg:wherePattern -> [ a lg:AntiJoinPattern ; lg:leftPattern [ a lg:JoinPa... |
| lg:demoRulePersistenceRendering | lg:QueryRendering | lg:wrapsQueryAST -> lg:demoRulePersistenceQuery ; rdf:value -> "SELECT $this WHERE { $this lg:viaRule ?r ; lg:toGra..." |
| lg:demoRulePersistenceConstraint | lg:SPARQLConstraint | lg:hasQueryAST -> lg:demoRulePersistenceQuery ; lg:hasQueryRendering -> lg:demoRulePersistenceRendering ; sh:select -> "SELECT $this WHERE { $this lg:viaRule ?r ; lg:toGra..." |

### Galois / polarity example
Resource blocks in stage: 19.
| Subject | rdf:type | Key outgoing links |
| --- | --- | --- |
| lg:NatSemiring | lg:Semiring | - |
| lg:BoolSemiring | lg:Semiring | - |
| lg:natSupportReflection | lg:SupportReflection | cat:cod -> lg:BoolSemiring ; cat:dom -> lg:NatSemiring |
| lg:eatWitnessDomain | lg:WitnessDomain | - |
| lg:eatPredicateDomain | lg:PredicateDomain | - |
| lg:eatIncidence | lg:IncidenceRelation | cat:cod -> lg:eatPredicateDomain ; cat:dom -> lg:eatWitnessDomain |
| lg:eatWitnessDerivation | lg:DerivationOperator | cat:cod -> lg:eatPredicateDomain ; cat:dom -> lg:eatWitnessDomain |
| lg:eatPredicateDerivation | lg:DerivationOperator | cat:cod -> lg:eatWitnessDomain ; cat:dom -> lg:eatPredicateDomain |
| lg:eatExtentClosure | lg:ClosureOperator | cat:cod -> lg:eatWitnessDomain ; cat:dom -> lg:eatWitnessDomain |
| lg:eatIntentClosure | lg:ClosureOperator | cat:cod -> lg:eatPredicateDomain ; cat:dom -> lg:eatPredicateDomain |
| lg:eatRulePolarity | lg:RulePolarity | lg:hasExtentClosure -> lg:eatExtentClosure ; lg:hasIncidenceRelation -> lg:eatIncidence ; lg:hasIntentClosure -> lg:eatIntentClosure |
| lg:eatMatrixFace | lg:MatrixPresentation | - |
| lg:eatKleisliFace | lg:KleisliPresentation | - |
| lg:eatPredicateFace | lg:PredicatePresentation | - |
| lg:eatSemanticAtlas | lg:SemanticAtlas | lg:hasKleisliPresentation -> lg:eatKleisliFace ; lg:hasMatrixPresentation -> lg:eatMatrixFace ; lg:hasPredicatePresentation -> lg:eatPredicateFace |
| lg:eatCategoricalClause | - | lg:hasSemanticAtlas -> lg:eatSemanticAtlas |
| lg:eatClosedExtent | lg:ClosedExtent | - |
| lg:eatClosedIntent | lg:ClosedIntent | - |
| lg:EatRule | lg:ClosedRuleCell | lg:hasClosedExtent -> lg:eatClosedExtent ; lg:hasClosedIntent -> lg:eatClosedIntent ; lg:hasPolarity -> lg:eatRulePolarity |

### Quotient recovery and concept lattice example
Resource blocks in stage: 26.
| Subject | rdf:type | Key outgoing links |
| --- | --- | --- |
| lg:eatSyntaxKernel | lg:KernelCongruence | - |
| lg:eatTypingKernel | lg:KernelCongruence | - |
| lg:eatSemanticKernel | lg:KernelCongruence | - |
| lg:eatExtentKernel | lg:KernelCongruence | - |
| lg:eatIntentKernel | lg:KernelCongruence | - |
| lg:eatSyntaxProjection | lg:QuotientProjection | cat:cod -> lg:eatSyntaxClause ; cat:dom -> lg:EatRule ; lg:quotientsBy -> lg:eatSyntaxKernel |
| lg:eatTypingProjection | lg:QuotientProjection | cat:cod -> lg:eatTypingClause ; cat:dom -> lg:EatRule ; lg:quotientsBy -> lg:eatTypingKernel |
| lg:eatSemanticProjection | lg:QuotientProjection | cat:cod -> lg:eatCategoricalClause ; cat:dom -> lg:EatRule ; lg:quotientsBy -> lg:eatSemanticKernel |
| lg:eatExtentProjection | lg:QuotientProjection | cat:cod -> lg:eatClosedExtent ; cat:dom -> lg:EatRule ; lg:quotientsBy -> lg:eatExtentKernel |
| lg:eatIntentProjection | lg:QuotientProjection | cat:cod -> lg:eatClosedIntent ; cat:dom -> lg:EatRule ; lg:quotientsBy -> lg:eatIntentKernel |
| lg:eatConceptOrder | lg:OrderRelation | - |
| lg:eatConceptMeet | lg:MeetOperation | - |
| lg:eatConceptJoin | lg:JoinOperation | - |
| lg:eatTopExtent | lg:ClosedExtent | - |
| lg:eatTopIntent | lg:ClosedIntent | - |
| lg:eatBottomExtent | lg:ClosedExtent | - |
| lg:eatBottomIntent | lg:ClosedIntent | - |
| lg:eatFormalConcept | lg:FormalConcept | lg:formalExtent -> lg:eatClosedExtent ; lg:formalIntent -> lg:eatClosedIntent |
| lg:eatTopConcept | lg:FormalConcept | lg:formalExtent -> lg:eatTopExtent ; lg:formalIntent -> lg:eatTopIntent |
| lg:eatBottomConcept | lg:FormalConcept | lg:formalExtent -> lg:eatBottomExtent ; lg:formalIntent -> lg:eatBottomIntent |
| lg:eatConceptLattice | lg:ConceptLattice | lg:generatedByPolarity -> lg:eatRulePolarity ; lg:hasBottomConcept -> lg:eatBottomConcept ; lg:hasFormalConcept -> lg:eatFormalConcept, lg:eatTopConcept |
| lg:eatRulePolarity | - | lg:inducesConceptLattice -> lg:eatConceptLattice |
| lg:eatFormalConcept | - | lg:memberOfLattice -> lg:eatConceptLattice |
| lg:eatTopConcept | - | lg:memberOfLattice -> lg:eatConceptLattice |
| lg:eatBottomConcept | - | lg:memberOfLattice -> lg:eatConceptLattice |
| lg:EatRule | - | lg:hasExtentProjection -> lg:eatExtentProjection ; lg:hasIntentProjection -> lg:eatIntentProjection ; lg:hasSemanticProjection -> lg:eatSemanticProjection |

### Theorem / proof-obligation example
Resource blocks in stage: 46.
| Subject | rdf:type | Key outgoing links |
| --- | --- | --- |
| lg:meetAssocSchema | lg:LawSchema | - |
| lg:joinAssocSchema | lg:LawSchema | - |
| lg:meetCommSchema | lg:LawSchema | - |
| lg:joinCommSchema | lg:LawSchema | - |
| lg:meetIdemSchema | lg:LawSchema | - |
| lg:joinIdemSchema | lg:LawSchema | - |
| lg:absorptionSchema | lg:LawSchema | - |
| lg:boundednessSchema | lg:LawSchema | - |
| lg:orderCompatibilitySchema | lg:LawSchema | - |
| lg:eatMeetAssociativityObligation | lg:MeetAssociativityObligation | lg:governsLattice -> lg:eatConceptLattice ; lg:governsMeetOperation -> lg:eatConceptMeet ; lg:statesLawSchema -> lg:meetAssocSchema |
| lg:eatJoinAssociativityObligation | lg:JoinAssociativityObligation | lg:governsJoinOperation -> lg:eatConceptJoin ; lg:governsLattice -> lg:eatConceptLattice ; lg:statesLawSchema -> lg:joinAssocSchema |
| lg:eatMeetCommutativityObligation | lg:MeetCommutativityObligation | lg:governsLattice -> lg:eatConceptLattice ; lg:governsMeetOperation -> lg:eatConceptMeet ; lg:statesLawSchema -> lg:meetCommSchema |
| lg:eatJoinCommutativityObligation | lg:JoinCommutativityObligation | lg:governsJoinOperation -> lg:eatConceptJoin ; lg:governsLattice -> lg:eatConceptLattice ; lg:statesLawSchema -> lg:joinCommSchema |
| lg:eatMeetIdempotenceObligation | lg:MeetIdempotenceObligation | lg:governsLattice -> lg:eatConceptLattice ; lg:governsMeetOperation -> lg:eatConceptMeet ; lg:statesLawSchema -> lg:meetIdemSchema |
| lg:eatJoinIdempotenceObligation | lg:JoinIdempotenceObligation | lg:governsJoinOperation -> lg:eatConceptJoin ; lg:governsLattice -> lg:eatConceptLattice ; lg:statesLawSchema -> lg:joinIdemSchema |
| lg:eatAbsorptionObligation | lg:AbsorptionObligation | lg:governsJoinOperation -> lg:eatConceptJoin ; lg:governsLattice -> lg:eatConceptLattice ; lg:governsMeetOperation -> lg:eatConceptMeet |
| lg:eatBoundednessObligation | lg:BoundednessObligation | lg:governsBottomConcept -> lg:eatBottomConcept ; lg:governsLattice -> lg:eatConceptLattice ; lg:governsTopConcept -> lg:eatTopConcept |
| lg:eatOrderCompatibilityObligation | lg:OrderCompatibilityObligation | lg:governsJoinOperation -> lg:eatConceptJoin ; lg:governsLattice -> lg:eatConceptLattice ; lg:governsMeetOperation -> lg:eatConceptMeet |
| lg:eatConceptLattice | - | lg:hasAbsorptionObligation -> lg:eatAbsorptionObligation ; lg:hasBoundednessObligation -> lg:eatBoundednessObligation ; lg:hasJoinAssociativityObligation -> lg:eatJoinAssociativityObligation |
| lg:eatMeetAssocWitness | lg:ProofWitness | lg:witnessesObligation -> lg:eatMeetAssociativityObligation |
| lg:eatJoinAssocWitness | lg:ProofWitness | lg:witnessesObligation -> lg:eatJoinAssociativityObligation |
| lg:eatAbsorptionWitness | lg:ProofWitness | lg:witnessesObligation -> lg:eatAbsorptionObligation |
| lg:eatMeetAssociativityObligation | - | lg:hasProofWitness -> lg:eatMeetAssocWitness |
| lg:eatJoinAssociativityObligation | - | lg:hasProofWitness -> lg:eatJoinAssocWitness |
| lg:eatAbsorptionObligation | - | lg:hasProofWitness -> lg:eatAbsorptionWitness |
| lg:quotientUniversalitySchema | lg:UniversalPropertySchema | - |
| lg:eatSyntaxFactorizationSchema | lg:FactorizationSchema | - |
| lg:eatSyntaxUniquenessSchema | lg:UniquenessSchema | - |
| lg:eatTypingFactorizationSchema | lg:FactorizationSchema | - |
| lg:eatTypingUniquenessSchema | lg:UniquenessSchema | - |
| lg:eatSemanticFactorizationSchema | lg:FactorizationSchema | - |
| lg:eatSemanticUniquenessSchema | lg:UniquenessSchema | - |
| lg:eatExtentFactorizationSchema | lg:FactorizationSchema | - |
| lg:eatExtentUniquenessSchema | lg:UniquenessSchema | - |
| lg:eatIntentFactorizationSchema | lg:FactorizationSchema | - |
| lg:eatIntentUniquenessSchema | lg:UniquenessSchema | - |
| lg:eatSyntaxUniversalityObligation | lg:QuotientUniversalityObligation | lg:governsKernelCongruence -> lg:eatSyntaxKernel ; lg:governsProjection -> lg:eatSyntaxProjection ; lg:governsQuotientFace -> lg:eatSyntaxClause |
| lg:eatTypingUniversalityObligation | lg:QuotientUniversalityObligation | lg:governsKernelCongruence -> lg:eatTypingKernel ; lg:governsProjection -> lg:eatTypingProjection ; lg:governsQuotientFace -> lg:eatTypingClause |
| lg:eatSemanticUniversalityObligation | lg:QuotientUniversalityObligation | lg:governsKernelCongruence -> lg:eatSemanticKernel ; lg:governsProjection -> lg:eatSemanticProjection ; lg:governsQuotientFace -> lg:eatCategoricalClause |
| lg:eatExtentUniversalityObligation | lg:QuotientUniversalityObligation | lg:governsKernelCongruence -> lg:eatExtentKernel ; lg:governsProjection -> lg:eatExtentProjection ; lg:governsQuotientFace -> lg:eatClosedExtent |
| lg:eatIntentUniversalityObligation | lg:QuotientUniversalityObligation | lg:governsKernelCongruence -> lg:eatIntentKernel ; lg:governsProjection -> lg:eatIntentProjection ; lg:governsQuotientFace -> lg:eatClosedIntent |
| lg:eatSyntaxProjection | - | lg:hasQuotientUniversalityObligation -> lg:eatSyntaxUniversalityObligation |
| lg:eatTypingProjection | - | lg:hasQuotientUniversalityObligation -> lg:eatTypingUniversalityObligation |
| lg:eatSemanticProjection | - | lg:hasQuotientUniversalityObligation -> lg:eatSemanticUniversalityObligation |
| lg:eatExtentProjection | - | lg:hasQuotientUniversalityObligation -> lg:eatExtentUniversalityObligation |
| lg:eatIntentProjection | - | lg:hasQuotientUniversalityObligation -> lg:eatIntentUniversalityObligation |

### Reflective SHACL boundary example
Resource blocks in stage: 20.
| Subject | rdf:type | Key outgoing links |
| --- | --- | --- |
| lg:coreConstraintProofCalculus | lg:InternalProofCalculus | - |
| lg:coreConstraintDenotationCategory | lg:SHACLDenotationCategory | - |
| lg:coreSupportedSHACLFragment | lg:WellFormedSHACLFragment | - |
| lg:coreObservationalEquivalence | lg:ObservationalEquivalence | - |
| lg:demoRulePersistenceDenotation | lg:ConstraintDenotation | cat:inCategory -> lg:coreConstraintDenotationCategory ; lg:canonicalInternalRepresentative -> lg:demoRulePersistenceProofTerm ; lg:representedByConstraintArtifact -> lg:demoRulePersistenceConstraint |
| lg:demoRulePersistenceProofTerm | lg:InternalProofTerm | cat:inCategory -> lg:coreConstraintProofCalculus ; lg:denotesConstraint -> lg:demoRulePersistenceDenotation |
| lg:demoRulePersistenceDerivation | lg:InternalDerivation | cat:inCategory -> lg:coreConstraintProofCalculus ; lg:derivesInternalProofTerm -> lg:demoRulePersistenceProofTerm |
| lg:coreConstraintProofCalculus | - | lg:hasInternalDerivation -> lg:demoRulePersistenceDerivation ; lg:hasInternalProofTerm -> lg:demoRulePersistenceProofTerm |
| lg:coreConstraintQuotient | lg:DenotationalQuotient | lg:byConstraintEquivalence -> lg:coreObservationalEquivalence ; lg:hasConstraintDenotation -> lg:demoRulePersistenceDenotation ; lg:hasDenotationCategory -> lg:coreConstraintDenotationCategory |
| lg:coreLoweringFunctor | lg:LoweringFunctor | cat:sourceCategory -> lg:coreConstraintProofCalculus ; cat:targetCategory -> lg:coreConstraintDenotationCategory |
| lg:coreReflectionFunctor | lg:ReflectionFunctor | cat:sourceCategory -> lg:coreConstraintDenotationCategory ; cat:targetCategory -> lg:coreConstraintProofCalculus |
| lg:coreLoweringSoundnessWitness | lg:ProofWitness | - |
| lg:coreSHACLSurjectivityWitness | lg:ProofWitness | - |
| lg:coreReflectiveQuotientWitness | lg:ProofWitness | - |
| lg:coreConservativeProjectionWitness | lg:ProofWitness | - |
| lg:coreReflectiveBoundary | lg:ReflectiveSHACLBoundary | lg:hasConservativeProjectionObligation -> lg:coreConservativeProjectionObligation ; lg:hasConstraintEquivalence -> lg:coreObservationalEquivalence ; lg:hasDenotationalQuotient -> lg:coreConstraintQuotient |
| lg:coreLoweringSoundnessObligation | lg:LoweringSoundnessObligation | lg:governsBoundary -> lg:coreReflectiveBoundary ; lg:governsConstraintEquivalence -> lg:coreObservationalEquivalence ; lg:governsDenotationalQuotient -> lg:coreConstraintQuotient |
| lg:coreSHACLSurjectivityObligation | lg:SHACLSurjectivityObligation | lg:governsBoundary -> lg:coreReflectiveBoundary ; lg:governsConstraintEquivalence -> lg:coreObservationalEquivalence ; lg:governsDenotationalQuotient -> lg:coreConstraintQuotient |
| lg:coreReflectiveQuotientObligation | lg:ReflectiveQuotientObligation | lg:governsBoundary -> lg:coreReflectiveBoundary ; lg:governsConstraintEquivalence -> lg:coreObservationalEquivalence ; lg:governsDenotationalQuotient -> lg:coreConstraintQuotient |
| lg:coreConservativeProjectionObligation | lg:ConservativeProjectionObligation | lg:governsBoundary -> lg:coreReflectiveBoundary ; lg:governsConstraintEquivalence -> lg:coreObservationalEquivalence ; lg:governsDenotationalQuotient -> lg:coreConstraintQuotient |

### Proof-carrying query / immutable history example
Resource blocks in stage: 25.
| Subject | rdf:type | Key outgoing links |
| --- | --- | --- |
| lg:historyLineage0 | lg:HistoryLineage | - |
| lg:historyState0 | lg:HistoryState | lg:hasHistoricalAssertion -> lg:histAssertViaRule, lg:histAssertToGrammar ; lg:stateDigest -> "sha256:history-state-0" |
| lg:historyLineage0 | - | lg:hasHistoryImmutabilityObligation -> lg:historyLineage0ImmutabilityObligation ; lg:hasHistoryState -> lg:historyState0 |
| lg:historyLineage0ImmutabilityObligation | lg:HistoryImmutabilityObligation | lg:governsHistoryLineage -> lg:historyLineage0 ; lg:hasProofWitness -> lg:historyLineage0ImmutabilityWitness |
| lg:historyLineage0ImmutabilityWitness | lg:ProofWitness | - |
| lg:histAssertViaRule | lg:HistoricalAssertion | lg:assertionObject -> lg:EatRule ; lg:assertionPredicate -> lg:viaRule ; lg:assertionSubject -> lg:eatEvolution |
| lg:histAssertToGrammar | lg:HistoricalAssertion | lg:assertionObject -> lg:G1 ; lg:assertionPredicate -> lg:toGrammarState ; lg:assertionSubject -> lg:eatEvolution |
| lg:histAssertHasRule | lg:HistoricalAssertion | lg:assertionObject -> lg:EatRule ; lg:assertionPredicate -> lg:hasRule ; lg:assertionSubject -> lg:G1 |
| lg:histAssertHasIdentifier | lg:HistoricalAssertion | lg:assertionObject -> lg:eatIdentifier ; lg:assertionPredicate -> lg:hasEmergentIdentifier ; lg:assertionSubject -> lg:G1 |
| lg:evolutionLookupVar | lg:QueryVariable | lg:variableName -> "r" |
| lg:evolutionLookupQuery | lg:SelectQuery | lg:projectsVariable -> lg:thisVar ; lg:wherePattern -> [ a lg:TriplePattern ; lg:subjectTerm lg:thisVar ; lg... |
| lg:evolutionLookupRendering | lg:QueryRendering | lg:wrapsQueryAST -> lg:evolutionLookupQuery ; rdf:value -> "SELECT $this WHERE { $this lg:viaRule ?r . }" |
| lg:evolutionLookupAlgebra | lg:ProjectAlgebraExpression | - |
| lg:evolutionLookupConstraint | lg:SPARQLConstraint | lg:hasQueryAST -> lg:evolutionLookupQuery ; lg:hasQueryRendering -> lg:evolutionLookupRendering ; sh:select -> "SELECT $this WHERE { $this lg:viaRule ?r . }" |
| lg:evolutionLookupDenotation | lg:ConstraintDenotation | cat:inCategory -> lg:coreConstraintDenotationCategory ; lg:canonicalInternalRepresentative -> lg:evolutionLookupProofQuery ; lg:representedByConstraintArtifact -> lg:evolutionLookupConstraint |
| lg:evolutionLookupResultSet | lg:QueryResultSet | - |
| lg:evolutionLookupMapping1 | lg:SolutionMapping | lg:hasBindingWitness -> lg:evolutionLookupBinding1 |
| lg:evolutionLookupBinding1 | lg:BindingWitness | lg:bindsValue -> lg:eatEvolution ; lg:bindsVariable -> lg:thisVar ; lg:justifiedByEvaluationStep -> lg:evolutionLookupStep1 |
| lg:evolutionLookupResultSet | - | lg:hasSolutionMapping -> lg:evolutionLookupMapping1 |
| lg:evolutionLookupStep1 | lg:EvaluationStep | lg:groundedInHistoricalAssertion -> lg:histAssertViaRule |
| lg:evolutionLookupTrace | lg:QueryEvaluationTrace | cat:inCategory -> lg:coreConstraintProofCalculus ; lg:derivesInternalProofTerm -> lg:evolutionLookupProofQuery ; lg:hasEvaluationStep -> lg:evolutionLookupStep1 |
| lg:evolutionLookupConstructiveWitness | lg:ConstructiveCarryingObligation | lg:governsHistoryLineage -> lg:historyLineage0 ; lg:governsHistoryState -> lg:historyState0 ; lg:governsProofCarryingQuery -> lg:evolutionLookupProofQuery |
| lg:evolutionLookupProofQuery | lg:ProofCarryingQuery | cat:inCategory -> lg:coreConstraintProofCalculus ; lg:belongsToHistoryLineage -> lg:historyLineage0 ; lg:denotesConstraint -> lg:evolutionLookupDenotation |
| lg:coreConstraintProofCalculus | - | lg:hasInternalDerivation -> lg:evolutionLookupTrace ; lg:hasInternalProofTerm -> lg:evolutionLookupProofQuery |
| lg:coreConstraintQuotient | - | lg:hasConstraintDenotation -> lg:evolutionLookupDenotation |

### Stamped history / wedge product / constructive empty-result example
Resource blocks in stage: 20.
| Subject | rdf:type | Key outgoing links |
| --- | --- | --- |
| lg:historyStamp0 | lg:HistoryStamp | lg:hasHistoryBoundaryCompletenessObligation -> lg:historyStamp0BoundaryCompletenessObligation ; lg:stampDigest -> "sha256:history-state-0" ; lg:stampedInHistoryLineage -> lg:historyLineage0 |
| lg:historyStamp0BoundaryCompletenessObligation | lg:HistoryBoundaryCompletenessObligation | lg:governsHistoryLineage -> lg:historyLineage0 ; lg:governsHistoryStamp -> lg:historyStamp0 ; lg:governsHistoryState -> lg:historyState0 |
| lg:evolutionLookupResultSet | - | lg:resultCardinality -> "1" |
| lg:evolutionLookupWedge | lg:QueryWedgeProduct | lg:hasWedgeHistoricalAssertion -> lg:histAssertViaRule ; lg:wedgesHistoryStamp -> lg:historyStamp0 ; lg:wedgesProofCarryingQuery -> lg:evolutionLookupProofQuery |
| lg:evolutionLookupProofQuery | - | lg:hasHistoryStamp -> lg:historyStamp0 ; lg:hasQueryWedgeProduct -> lg:evolutionLookupWedge |
| lg:missingRule | lg:AugmentedRule | - |
| lg:emptyViaMissingRuleQuery | lg:SelectQuery | lg:projectsVariable -> lg:thisVar ; lg:wherePattern -> [ a lg:TriplePattern ; lg:subjectTerm lg:thisVar ; lg... |
| lg:emptyViaMissingRuleRendering | lg:QueryRendering | lg:wrapsQueryAST -> lg:emptyViaMissingRuleQuery ; rdf:value -> "SELECT $this WHERE { $this lg:viaRule lg:missingRul..." |
| lg:emptyViaMissingRuleAlgebra | lg:ProjectAlgebraExpression | - |
| lg:emptyViaMissingRuleConstraint | lg:SPARQLConstraint | lg:hasQueryAST -> lg:emptyViaMissingRuleQuery ; lg:hasQueryRendering -> lg:emptyViaMissingRuleRendering ; sh:select -> "SELECT $this WHERE { $this lg:viaRule lg:missingRul..." |
| lg:emptyViaMissingRuleDenotation | lg:ConstraintDenotation | cat:inCategory -> lg:coreConstraintDenotationCategory ; lg:canonicalInternalRepresentative -> lg:emptyViaMissingRuleProofQuery ; lg:representedByConstraintArtifact -> lg:emptyViaMissingRuleConstraint |
| lg:emptyViaMissingRuleResultSet | lg:EmptyQueryResultSet | lg:resultCardinality -> "0" |
| lg:emptyViaMissingRuleWedge | lg:EmptyWedgeProduct | lg:wedgesHistoryStamp -> lg:historyStamp0 ; lg:wedgesProofCarryingQuery -> lg:emptyViaMissingRuleProofQuery |
| lg:emptyViaMissingRuleAbsenceStep | lg:AbsenceEvaluationStep | lg:groundedInHistoryStamp -> lg:historyStamp0 ; lg:verifiesEmptyWedgeProduct -> lg:emptyViaMissingRuleWedge |
| lg:emptyViaMissingRuleTrace | lg:AbsenceEvaluationTrace | cat:inCategory -> lg:coreConstraintProofCalculus ; lg:derivesInternalProofTerm -> lg:emptyViaMissingRuleProofQuery ; lg:hasEvaluationStep -> lg:emptyViaMissingRuleAbsenceStep |
| lg:emptyViaMissingRuleWitness | lg:EmptyResultWitness | lg:witnessesAbsenceTrace -> lg:emptyViaMissingRuleTrace ; lg:witnessesEmptyResultSet -> lg:emptyViaMissingRuleResultSet ; lg:witnessesEmptyWedgeProduct -> lg:emptyViaMissingRuleWedge |
| lg:emptyViaMissingRuleConstructiveObligation | lg:EmptyResultConstructivityObligation | lg:governsEmptyResultWitness -> lg:emptyViaMissingRuleWitness ; lg:governsHistoryLineage -> lg:historyLineage0 ; lg:governsHistoryStamp -> lg:historyStamp0 |
| lg:emptyViaMissingRuleProofQuery | lg:EmptyResultProofQuery | cat:inCategory -> lg:coreConstraintProofCalculus ; lg:belongsToHistoryLineage -> lg:historyLineage0 ; lg:denotesConstraint -> lg:emptyViaMissingRuleDenotation |
| lg:coreConstraintProofCalculus | - | lg:hasInternalDerivation -> lg:emptyViaMissingRuleTrace ; lg:hasInternalProofTerm -> lg:emptyViaMissingRuleProofQuery |
| lg:coreConstraintQuotient | - | lg:hasConstraintDenotation -> lg:emptyViaMissingRuleDenotation |

### Adjoint EXISTS / NOT EXISTS over stamped-history wedges example
Resource blocks in stage: 51.
| Subject | rdf:type | Key outgoing links |
| --- | --- | --- |
| lg:outerThisContext | lg:QueryContext | rdfs:label -> "Outer this-context" |
| lg:outerThisWitnessContext | lg:QueryContext | rdfs:label -> "Outer this-plus-witness context" |
| lg:viaRuleWitnessProjection | lg:ContextProjection | cat:cod -> lg:outerThisContext ; cat:dom -> lg:outerThisWitnessContext ; rdfs:label -> "Projection forgetting ?r" |
| lg:viaRuleReindexing | lg:ReindexingOperator | cat:sourceCategory -> lg:coreConstraintProofCalculus ; cat:targetCategory -> lg:coreConstraintProofCalculus ; rdfs:label -> "Reindexing along viaRule witness projection" |
| lg:viaRuleExistsOperator | lg:ExistentialImageOperator | rdfs:label -> "Existential image along viaRule witness projection" |
| lg:viaRuleAdjointPair | lg:AdjointPair | lg:alongContextProjection -> lg:viaRuleWitnessProjection ; lg:hasLeftAdjointOperator -> lg:viaRuleExistsOperator ; lg:hasRightAdjointOperator -> lg:viaRuleReindexing |
| lg:existsViaRuleQuery | lg:SelectQuery | lg:projectsVariable -> lg:thisVar ; lg:wherePattern -> [ a lg:TriplePattern ; lg:subjectTerm lg:thisVar ; lg... |
| lg:existsViaRuleRendering | lg:QueryRendering | lg:wrapsQueryAST -> lg:existsViaRuleQuery ; rdf:value -> "SELECT $this WHERE { $this lg:viaRule ?r . }" |
| lg:existsViaRuleAlgebra | lg:ExistsAlgebraExpression | - |
| lg:existsViaRuleConstraint | lg:SPARQLConstraint | lg:hasQueryAST -> lg:existsViaRuleQuery ; lg:hasQueryRendering -> lg:existsViaRuleRendering ; sh:select -> "SELECT $this WHERE { $this lg:viaRule ?r . }" |
| lg:existsViaRuleDenotation | lg:ConstraintDenotation | cat:inCategory -> lg:coreConstraintDenotationCategory ; lg:canonicalInternalRepresentative -> lg:existsViaRuleProofQuery ; lg:representedByConstraintArtifact -> lg:existsViaRuleConstraint |
| lg:existsViaRuleResultSet | lg:QueryResultSet | lg:hasSolutionMapping -> lg:existsViaRuleMapping1 ; lg:resultCardinality -> "1" |
| lg:existsViaRuleBinding1 | lg:BindingWitness | lg:bindsValue -> lg:eatEvolution ; lg:bindsVariable -> lg:thisVar ; lg:justifiedByEvaluationStep -> lg:existsViaRuleStep1 |
| lg:existsViaRuleMapping1 | lg:SolutionMapping | lg:hasBindingWitness -> lg:existsViaRuleBinding1 |
| lg:existsViaRuleStep1 | lg:EvaluationStep | lg:groundedInHistoricalAssertion -> lg:histAssertViaRule |
| lg:existsViaRuleTrace | lg:QueryEvaluationTrace | cat:inCategory -> lg:coreConstraintProofCalculus ; lg:derivesInternalProofTerm -> lg:existsViaRuleProofQuery ; lg:hasEvaluationStep -> lg:existsViaRuleStep1 |
| lg:existsViaRuleWedge | lg:QueryWedgeProduct | lg:hasWedgeHistoricalAssertion -> lg:histAssertViaRule ; lg:wedgesHistoryStamp -> lg:historyStamp0 ; lg:wedgesProofCarryingQuery -> lg:existsViaRuleProofQuery |
| lg:existsViaRuleImage | lg:ExistentialImage | lg:alongWitnessProjection -> lg:viaRuleWitnessProjection ; lg:derivedFromQueryWedgeProduct -> lg:existsViaRuleWedge |
| lg:existsViaRuleConstructiveObligation | lg:ConstructiveCarryingObligation | lg:governsHistoryLineage -> lg:historyLineage0 ; lg:governsHistoryState -> lg:historyState0 ; lg:governsProofCarryingQuery -> lg:existsViaRuleProofQuery |
| lg:existsViaRuleAdjunctionObligation | lg:ExistentialAdjunctionObligation | lg:governsAdjointPair -> lg:viaRuleAdjointPair ; lg:governsContextProjection -> lg:viaRuleWitnessProjection ; lg:governsExistentialImage -> lg:existsViaRuleImage |
| lg:existsViaRuleProofQuery | lg:ExistsWitnessedQuery | cat:inCategory -> lg:coreConstraintProofCalculus ; lg:belongsToHistoryLineage -> lg:historyLineage0 ; lg:denotesConstraint -> lg:existsViaRuleDenotation |
| lg:coreConstraintProofCalculus | - | lg:hasInternalDerivation -> lg:existsViaRuleTrace ; lg:hasInternalProofTerm -> lg:existsViaRuleProofQuery |
| lg:coreConstraintQuotient | - | lg:hasConstraintDenotation -> lg:existsViaRuleDenotation |
| lg:notExistsMissingRuleLeftPattern | lg:TriplePattern | lg:objectTerm -> lg:rVar ; lg:predicateTerm -> lg:viaRule ; lg:subjectTerm -> lg:thisVar |
| lg:notExistsMissingRuleRightPattern | lg:TriplePattern | lg:objectTerm -> lg:missingRule ; lg:predicateTerm -> lg:viaRule ; lg:subjectTerm -> lg:thisVar |
| lg:notExistsMissingRuleAntiJoin | lg:AntiJoinPattern | lg:leftPattern -> lg:notExistsMissingRuleLeftPattern ; lg:rightPattern -> lg:notExistsMissingRuleRightPattern |
| lg:notExistsMissingRuleQuery | lg:SelectQuery | lg:projectsVariable -> lg:thisVar ; lg:wherePattern -> lg:notExistsMissingRuleAntiJoin |
| lg:notExistsMissingRuleRendering | lg:QueryRendering | lg:wrapsQueryAST -> lg:notExistsMissingRuleQuery ; rdf:value -> "SELECT $this WHERE { $this lg:viaRule ?r . FILTER N..." |
| lg:notExistsMissingRuleAlgebra | lg:NotExistsAlgebraExpression | - |
| lg:notExistsMissingRuleConstraint | lg:SPARQLConstraint | lg:hasQueryAST -> lg:notExistsMissingRuleQuery ; lg:hasQueryRendering -> lg:notExistsMissingRuleRendering ; sh:select -> "SELECT $this WHERE { $this lg:viaRule ?r . FILTER N..." |
| lg:notExistsMissingRuleDenotation | lg:ConstraintDenotation | cat:inCategory -> lg:coreConstraintDenotationCategory ; lg:canonicalInternalRepresentative -> lg:notExistsMissingRuleProofQuery ; lg:representedByConstraintArtifact -> lg:notExistsMissingRuleConstraint |
| lg:notExistsMissingRuleResultSet | lg:QueryResultSet | lg:hasSolutionMapping -> lg:notExistsMissingRuleMapping1 ; lg:resultCardinality -> "1" |
| lg:notExistsMissingRuleBinding1 | lg:BindingWitness | lg:bindsValue -> lg:eatEvolution ; lg:bindsVariable -> lg:thisVar ; lg:justifiedByEvaluationStep -> lg:notExistsMissingRuleStep1 |
| lg:notExistsMissingRuleMapping1 | lg:SolutionMapping | lg:hasBindingWitness -> lg:notExistsMissingRuleBinding1 |
| lg:notExistsMissingRuleStep1 | lg:EvaluationStep | lg:groundedInHistoricalAssertion -> lg:histAssertViaRule |
| lg:notExistsMissingRuleTrace | lg:QueryEvaluationTrace | cat:inCategory -> lg:coreConstraintProofCalculus ; lg:derivesInternalProofTerm -> lg:notExistsMissingRuleProofQuery ; lg:hasEvaluationStep -> lg:notExistsMissingRuleStep1 |
| lg:notExistsMissingRuleWedge | lg:QueryWedgeProduct | lg:hasWedgeHistoricalAssertion -> lg:histAssertViaRule ; lg:wedgesHistoryStamp -> lg:historyStamp0 ; lg:wedgesProofCarryingQuery -> lg:notExistsMissingRuleProofQuery |
| lg:missingRuleSupportImage | lg:SupportReflectedImage | lg:reflectedFromExistentialImage -> lg:emptyViaMissingRuleImage |
| lg:missingRuleNegatedImage | lg:NegatedExistentialImage | lg:negatesSupportReflectedImage -> lg:missingRuleSupportImage |
| lg:missingRuleSupportReflectionOperator | lg:SupportReflectionOperator | rdfs:label -> "Support reflection for missingRule witness image" |
| lg:missingRuleNegationOperator | lg:NegationOperator | rdfs:label -> "Negation of support-reflected missingRule image" |
| lg:notExistsMissingRuleConstructiveObligation | lg:ConstructiveCarryingObligation | lg:governsHistoryLineage -> lg:historyLineage0 ; lg:governsHistoryState -> lg:historyState0 ; lg:governsProofCarryingQuery -> lg:notExistsMissingRuleProofQuery |
| lg:notExistsMissingRuleSupportReflectionObligation | lg:SupportReflectionObligation | lg:governsExistentialImage -> lg:emptyViaMissingRuleImage ; lg:governsHistoryStamp -> lg:historyStamp0 ; lg:governsProofCarryingQuery -> lg:notExistsMissingRuleProofQuery |
| lg:notExistsMissingRuleAdjointDerivationObligation | lg:NotExistsAdjointDerivationObligation | lg:governsHistoryStamp -> lg:historyStamp0 ; lg:governsNegatedExistentialImage -> lg:missingRuleNegatedImage ; lg:governsProofCarryingQuery -> lg:notExistsMissingRuleProofQuery |
| lg:notExistsMissingRuleProofQuery | lg:NotExistsWitnessedQuery | cat:inCategory -> lg:coreConstraintProofCalculus ; lg:belongsToHistoryLineage -> lg:historyLineage0 ; lg:denotesConstraint -> lg:notExistsMissingRuleDenotation |
| lg:coreConstraintProofCalculus | - | lg:hasInternalDerivation -> lg:notExistsMissingRuleTrace ; lg:hasInternalProofTerm -> lg:notExistsMissingRuleProofQuery |
| lg:coreConstraintQuotient | - | lg:hasConstraintDenotation -> lg:notExistsMissingRuleDenotation |
| lg:emptyViaMissingRuleImage | lg:ExistentialImage | lg:alongWitnessProjection -> lg:viaRuleWitnessProjection ; lg:derivedFromQueryWedgeProduct -> lg:emptyViaMissingRuleWedge |
| lg:emptyViaMissingRuleAdjointPair | lg:AdjointPair | lg:alongContextProjection -> lg:viaRuleWitnessProjection ; lg:hasLeftAdjointOperator -> lg:viaRuleExistsOperator ; lg:hasRightAdjointOperator -> lg:viaRuleReindexing |
| lg:emptyViaMissingRuleAdjunctionObligation | lg:ExistentialAdjunctionObligation | lg:governsAdjointPair -> lg:emptyViaMissingRuleAdjointPair ; lg:governsContextProjection -> lg:viaRuleWitnessProjection ; lg:governsExistentialImage -> lg:emptyViaMissingRuleImage |
| lg:emptyViaMissingRuleProofQuery | - | lg:hasAdjointPair -> lg:emptyViaMissingRuleAdjointPair ; lg:hasContextProjection -> lg:viaRuleWitnessProjection ; lg:hasExistentialAdjunctionObligation -> lg:emptyViaMissingRuleAdjunctionObligation |

### Explicit semantic policy choices for NOT EXISTS
Resource blocks in stage: 16.
| Subject | rdf:type | Key outgoing links |
| --- | --- | --- |
| lg:booleanTruthStructure | lg:BooleanTruthValueStructure | rdfs:label -> "Boolean truth-value structure for support-reflected..." |
| lg:missingRuleBooleanSupportPolicy | lg:BooleanSupportReflectionPolicy | lg:declaresSupportTruthValueStructure -> lg:booleanTruthStructure ; lg:selectsSupportReflectionOperator -> lg:missingRuleSupportReflectionOperator |
| lg:missingRuleBooleanNegationPolicy | lg:BooleanNegationPolicy | lg:declaresNegationTruthValueStructure -> lg:booleanTruthStructure ; lg:requiresSupportPolicy -> lg:missingRuleBooleanSupportPolicy ; lg:selectsNegationOperator -> lg:missingRuleNegationOperator |
| lg:notExistsMissingRulePolicyBundle | lg:SemanticPolicyBundle | lg:hasNegationPolicy -> lg:missingRuleBooleanNegationPolicy ; lg:hasSupportPolicy -> lg:missingRuleBooleanSupportPolicy ; lg:hasTruthValueStructure -> lg:booleanTruthStructure |
| lg:missingRuleSupportImage | - | lg:interpretedInTruthValueStructure -> lg:booleanTruthStructure |
| lg:missingRuleNegatedImage | - | lg:interpretedInTruthValueStructure -> lg:booleanTruthStructure |
| lg:notExistsMissingRulePolicyCoherenceObligation | lg:SemanticPolicyCoherenceObligation | lg:governsHistoryStamp -> lg:historyStamp0 ; lg:governsNegatedExistentialImage -> lg:missingRuleNegatedImage ; lg:governsNegationPolicy -> lg:missingRuleBooleanNegationPolicy |
| lg:notExistsMissingRuleBooleanSemanticsObligation | lg:BooleanNegationSemanticsObligation | lg:governsHistoryStamp -> lg:historyStamp0 ; lg:governsNegatedExistentialImage -> lg:missingRuleNegatedImage ; lg:governsNegationPolicy -> lg:missingRuleBooleanNegationPolicy |
| lg:notExistsMissingRuleProofQuery | - | lg:hasNegationPolicySemanticsObligation -> lg:notExistsMissingRuleBooleanSemanticsObligation ; lg:hasSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:hasSemanticPolicyCoherenceObligation -> lg:notExistsMissingRulePolicyCoherenceObligation |
| lg:heytingTruthStructure | lg:HeytingTruthValueStructure | rdfs:label -> "Prototype Heyting truth-value structure" |
| lg:prototypeHeytingSupportPolicy | lg:HeytingSupportReflectionPolicy | lg:declaresSupportTruthValueStructure -> lg:heytingTruthStructure ; lg:selectsSupportReflectionOperator -> lg:missingRuleSupportReflectionOperator |
| lg:prototypeHeytingNegationPolicy | lg:HeytingNegationPolicy | lg:declaresNegationTruthValueStructure -> lg:heytingTruthStructure ; lg:requiresSupportPolicy -> lg:prototypeHeytingSupportPolicy ; lg:selectsNegationOperator -> lg:missingRuleNegationOperator |
| lg:residuatedTruthStructure | lg:ResiduatedTruthValueStructure | rdfs:label -> "Prototype residuated truth-value structure" |
| lg:prototypeResiduatedNegationPolicy | lg:ResiduatedNegationPolicy | lg:declaresNegationTruthValueStructure -> lg:residuatedTruthStructure ; lg:requiresSupportPolicy -> lg:missingRuleBooleanSupportPolicy ; lg:selectsNegationOperator -> lg:missingRuleNegationOperator |
| lg:girardTruthStructure | lg:GirardTruthValueStructure | rdfs:label -> "Prototype Girard truth-value structure" |
| lg:prototypeGirardNegationPolicy | lg:GirardNegationPolicy | lg:declaresNegationTruthValueStructure -> lg:girardTruthStructure ; lg:requiresSupportPolicy -> lg:missingRuleBooleanSupportPolicy ; lg:selectsNegationOperator -> lg:missingRuleNegationOperator |

### Policy-indexed proof calculus example
Resource blocks in stage: 12.
| Subject | rdf:type | Key outgoing links |
| --- | --- | --- |
| lg:coreConstraintProofCalculus | lg:PolicyIndexedProofCalculus | lg:registersPolicyIndexedDerivation -> lg:notExistsMissingRulePolicyDerivation ; lg:registersPolicyIndexedProofTerm -> lg:notExistsMissingRuleProofQuery |
| lg:historyStampProofConstructor0 | lg:HistoryStampProofConstructor | - |
| lg:existentialImageProofConstructor0 | lg:ExistentialImageProofConstructor | - |
| lg:supportReflectionProofConstructor0 | lg:SupportReflectionProofConstructor | - |
| lg:negationProofConstructor0 | lg:NegationProofConstructor | - |
| lg:notExistsMissingRulePolicyJudgment | lg:PolicyScopedJudgment | lg:judgmentAboutQuery -> lg:notExistsMissingRuleProofQuery ; lg:judgmentTruthValueStructure -> lg:booleanTruthStructure ; lg:judgmentUnderSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle |
| lg:notExistsMissingRulePolicyCertificate | lg:PolicyCertificate | lg:certifiesHistoryStamp -> lg:historyStamp0 ; lg:certifiesNegationPolicy -> lg:missingRuleBooleanNegationPolicy ; lg:certifiesProofCarryingQuery -> lg:notExistsMissingRuleProofQuery |
| lg:notExistsMissingRulePolicyDerivation | lg:PolicyIndexedDerivation | cat:inCategory -> lg:coreConstraintProofCalculus ; lg:concludesPolicyScopedJudgment -> lg:notExistsMissingRulePolicyJudgment ; lg:derivationUnderNegationPolicy -> lg:missingRuleBooleanNegationPolicy |
| lg:notExistsMissingRulePolicyDerivationObligation | lg:PolicyIndexedConstructiveDerivationObligation | lg:governsBoundary -> lg:coreReflectiveBoundary ; lg:governsHistoryStamp -> lg:historyStamp0 ; lg:governsNegationPolicy -> lg:missingRuleBooleanNegationPolicy |
| lg:notExistsMissingRulePolicyPreservingLoweringObligation | lg:PolicyPreservingLoweringObligation | lg:governsBoundary -> lg:coreReflectiveBoundary ; lg:governsConstraintEquivalence -> lg:coreObservationalEquivalence ; lg:governsDenotationalQuotient -> lg:coreConstraintQuotient |
| lg:notExistsMissingRulePolicyReflectiveCompatibilityObligation | lg:PolicyReflectiveCompatibilityObligation | lg:governsBoundary -> lg:coreReflectiveBoundary ; lg:governsConstraintEquivalence -> lg:coreObservationalEquivalence ; lg:governsDenotationalQuotient -> lg:coreConstraintQuotient |
| lg:notExistsMissingRuleProofQuery | lg:PolicyIndexedNotExistsProofQuery | lg:derivedUnderSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:hasPolicyCertificate -> lg:notExistsMissingRulePolicyCertificate ; lg:hasPolicyDerivation -> lg:notExistsMissingRulePolicyDerivation |

### Compositional proof-term witness for NOT EXISTS
Resource blocks in stage: 16.
| Subject | rdf:type | Key outgoing links |
| --- | --- | --- |
| lg:sequentialCompositionProofConstructor0 | lg:ProofConstructor | rdfs:label -> "Sequential proof composition constructor" |
| lg:historyStampConstructorTypingObligation0 | lg:ConstructorTypingObligation | lg:governsHistoryStamp -> lg:historyStamp0 ; lg:governsProofConstructor -> lg:historyStampProofConstructor0 ; lg:governsSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle |
| lg:existentialImageConstructorTypingObligation0 | lg:ConstructorTypingObligation | lg:governsExistentialImage -> lg:emptyViaMissingRuleImage ; lg:governsProofConstructor -> lg:existentialImageProofConstructor0 ; lg:governsQueryWedgeProduct -> lg:emptyViaMissingRuleWedge |
| lg:supportReflectionConstructorTypingObligation0 | lg:ConstructorTypingObligation | lg:governsProofConstructor -> lg:supportReflectionProofConstructor0 ; lg:governsSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:governsSupportPolicy -> lg:missingRuleBooleanSupportPolicy |
| lg:negationConstructorTypingObligation0 | lg:ConstructorTypingObligation | lg:governsNegatedExistentialImage -> lg:missingRuleNegatedImage ; lg:governsNegationPolicy -> lg:missingRuleBooleanNegationPolicy ; lg:governsProofConstructor -> lg:negationProofConstructor0 |
| lg:proofCompositionObligation0 | lg:ProofTermCompositionObligation | lg:governsProofConstructor -> lg:sequentialCompositionProofConstructor0 ; lg:governsSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:governsSourceProofTerm -> lg:missingRuleHistoryStampTerm |
| lg:normalizationSoundnessObligation0 | lg:NormalizationSoundnessObligation | lg:governsSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:governsSourceProofTerm -> lg:notExistsMissingRuleSequentialProof ; lg:governsTargetProofTerm -> lg:notExistsMissingRuleProofQuery |
| lg:proofTermAdequacyObligation0 | lg:ProofTermAdequacyObligation | lg:governsPolicyIndexedDerivation -> lg:notExistsMissingRulePolicyDerivation ; lg:governsPolicyIndexedProofTerm -> lg:notExistsMissingRuleProofQuery ; lg:governsProofTerm -> lg:notExistsMissingRuleSequentialProof |
| lg:missingRuleHistoryStampTerm | lg:HistoryStampIntroTerm, lg:AtomicProofTerm | lg:denotesConstraint -> lg:emptyViaMissingRuleDenotation ; lg:derivedUnderSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:hasConstructorTypingObligation -> lg:historyStampConstructorTypingObligation0 |
| lg:emptyViaMissingRuleExistsTerm | lg:ExistentialImageIntroTerm | lg:denotesConstraint -> lg:emptyViaMissingRuleDenotation ; lg:derivedUnderSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:hasConstructorTypingObligation -> lg:existentialImageConstructorTypingObligation0 |
| lg:missingRuleSupportReflectionTerm | lg:SupportReflectionIntroTerm | lg:denotesConstraint -> lg:notExistsMissingRuleDenotation ; lg:derivedUnderSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:hasConstructorTypingObligation -> lg:supportReflectionConstructorTypingObligation0 |
| lg:notExistsMissingRuleNormalizationWitness | lg:NormalizationWitness | lg:normalizesSourceProofTerm -> lg:notExistsMissingRuleSequentialProof ; lg:normalizesTargetProofTerm -> lg:notExistsMissingRuleProofQuery |
| lg:notExistsMissingRuleSequentialProof | lg:SequentialProofComposition | lg:denotesConstraint -> lg:notExistsMissingRuleDenotation ; lg:derivedUnderSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:hasLeftProofTerm -> lg:missingRuleHistoryStampTerm |
| lg:notExistsMissingRulePolicyDerivation | - | lg:realizedByProofTerm -> lg:notExistsMissingRuleSequentialProof |
| lg:notExistsMissingRuleProofQuery | lg:CanonicalProofTerm, lg:NegationIntroTerm | lg:concludesPolicyScopedJudgment -> lg:notExistsMissingRulePolicyJudgment ; lg:derivedUnderSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:hasConstructorTypingObligation -> lg:negationConstructorTypingObligation0 |
| lg:coreConstraintProofCalculus | - | lg:registersPolicyIndexedDerivation -> lg:notExistsMissingRulePolicyDerivation ; lg:registersPolicyIndexedProofTerm -> lg:notExistsMissingRuleProofQuery, lg:missingRuleHistoryStampTerm |

### Substitution, cut-elimination, and shared-subproof reuse example
Resource blocks in stage: 24.
| Subject | rdf:type | Key outgoing links |
| --- | --- | --- |
| lg:substitutionProofConstructor0 | lg:ProofConstructor | rdfs:label -> "Substitution constructor" |
| lg:cutProofConstructor0 | lg:ProofConstructor | rdfs:label -> "Cut constructor" |
| lg:coreSubproofReuseCategory | lg:SubproofReuseCategory | - |
| lg:notExistsProofVar | lg:ProofVariable | lg:denotesConstraint -> lg:notExistsMissingRuleDenotation ; lg:derivedUnderSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:proofVariableName -> "p_support" |
| lg:notExistsProofContext | lg:ProofContext | lg:bindsProofVariable -> lg:notExistsProofVar |
| lg:notExistsSupportAssumption | lg:ProofAssumption | lg:assumesProofTerm -> lg:missingRuleSupportReflectionTerm ; lg:assumesProofVariable -> lg:notExistsProofVar ; lg:inProofContext -> lg:notExistsProofContext |
| lg:notExistsProofContext | - | lg:hasProofAssumption -> lg:notExistsSupportAssumption |
| lg:notExistsNegationSchemaTerm | lg:UnaryProofConstructorApplication, lg:CompositeProofTerm | lg:denotesConstraint -> lg:notExistsMissingRuleDenotation ; lg:derivedUnderSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:hasPremiseProofTerm -> lg:notExistsProofVar |
| lg:notExistsSubstitutionWitness0 | lg:SubstitutionWitness | lg:substitutesReplacementTerm -> lg:missingRuleSupportReflectionTerm ; lg:substitutesSourceProofTerm -> lg:notExistsNegationSchemaTerm ; lg:substitutesTargetProofTerm -> lg:notExistsMissingRuleProofQuery |
| lg:notExistsSubstitutionSoundnessObligation0 | lg:SubstitutionSoundnessObligation | lg:governsProofVariable -> lg:notExistsProofVar ; lg:governsSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:governsSourceProofTerm -> lg:notExistsNegationSchemaTerm |
| lg:notExistsSubstitutionCompatibilityObligation0 | lg:SubstitutionCompatibilityObligation | lg:governsProofAssumption -> lg:notExistsSupportAssumption ; lg:governsProofContext -> lg:notExistsProofContext ; lg:governsSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle |
| lg:notExistsNegationSubstitution0 | lg:SubstitutionApplication | lg:denotesConstraint -> lg:notExistsMissingRuleDenotation ; lg:derivedUnderSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:hasSubstitutionCompatibilityObligation -> lg:notExistsSubstitutionCompatibilityObligation0 |
| lg:notExistsCutEliminationWitness0 | lg:CutEliminationWitness | lg:cutEliminatesToProofTerm -> lg:notExistsMissingRuleProofQuery ; lg:eliminatesCutSourceProofTerm -> lg:notExistsCutApplication0 ; lg:normalizesSourceProofTerm -> lg:notExistsCutApplication0 |
| lg:notExistsCutAdmissibilityObligation0 | lg:CutAdmissibilityObligation | lg:governsCutApplication -> lg:notExistsCutApplication0 ; lg:governsProofAssumption -> lg:notExistsSupportAssumption ; lg:governsProofVariable -> lg:notExistsProofVar |
| lg:notExistsCutEliminationObligation0 | lg:CutEliminationObligation | lg:governsCutApplication -> lg:notExistsCutApplication0 ; lg:governsSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:governsSourceProofTerm -> lg:notExistsCutApplication0 |
| lg:notExistsCutApplication0 | lg:CutApplication | lg:cutsOnProofAssumption -> lg:notExistsSupportAssumption ; lg:cutsOnProofVariable -> lg:notExistsProofVar ; lg:denotesConstraint -> lg:notExistsMissingRuleDenotation |
| lg:missingRuleSupportReuseObligation0 | lg:SharedSubproofReuseObligation | lg:governsReuseCategory -> lg:coreSubproofReuseCategory ; lg:governsSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:governsSharedSubproof -> lg:missingRuleSupportReflectionTerm |
| lg:coreSubproofFunctorialityObligation0 | lg:FunctorialSubproofReuseObligation | lg:governsReuseCategory -> lg:coreSubproofReuseCategory ; lg:governsSemanticPolicyBundle -> lg:notExistsMissingRulePolicyBundle ; lg:hasProofWitness -> lg:supportReuseMorphism0 |
| lg:missingRuleSupportReflectionTerm | lg:SharedSubproof | lg:hasSubproofReuseObligation -> lg:missingRuleSupportReuseObligation0 ; lg:registeredInReuseCategory -> lg:coreSubproofReuseCategory |
| lg:supportReuseMorphism0 | lg:SubproofReuseMorphism | cat:cod -> lg:missingRuleSupportReflectionTerm ; cat:dom -> lg:missingRuleSupportReflectionTerm ; cat:inCategory -> lg:coreSubproofReuseCategory |
| lg:coreSubproofReuseFunctor0 | lg:SubproofReuseFunctor | cat:mapsMorphism -> lg:supportReuseMorphism0 ; cat:mapsObject -> lg:missingRuleSupportReflectionTerm ; cat:sourceCategory -> lg:coreSubproofReuseCategory |
| lg:coreSubproofReuseCategory | - | lg:hasFunctorialSubproofReuseObligation -> lg:coreSubproofFunctorialityObligation0 ; lg:hasSubproofReuseMorphism -> lg:supportReuseMorphism0 |
| lg:notExistsCutApplication0 | - | lg:hasSharedSubproof -> lg:missingRuleSupportReflectionTerm |
| lg:notExistsMissingRuleProofQuery | lg:CutFreeProofTerm | lg:hasSharedSubproof -> lg:missingRuleSupportReflectionTerm |

## Query Renderings

### lg:demoRulePersistenceRendering
- `wrapsQueryAST`: `lg:demoRulePersistenceQuery`

```sparql
    SELECT $this
    WHERE {
      $this lg:viaRule ?r ;
            lg:toGrammarState ?g2 .
      FILTER NOT EXISTS { ?g2 lg:hasRule ?r . }
    }
  
```

### lg:emptyViaMissingRuleRendering
- `wrapsQueryAST`: `lg:emptyViaMissingRuleQuery`

```sparql
    SELECT $this
    WHERE {
      $this lg:viaRule lg:missingRule .
    }
  
```

### lg:evolutionLookupRendering
- `wrapsQueryAST`: `lg:evolutionLookupQuery`

```sparql
    SELECT $this
    WHERE {
      $this lg:viaRule ?r .
    }
  
```

### lg:existsViaRuleRendering
- `wrapsQueryAST`: `lg:existsViaRuleQuery`

```sparql
    SELECT $this
    WHERE {
      $this lg:viaRule ?r .
    }
  
```

### lg:notExistsMissingRuleRendering
- `wrapsQueryAST`: `lg:notExistsMissingRuleQuery`

```sparql
    SELECT $this
    WHERE {
      $this lg:viaRule ?r .
      FILTER NOT EXISTS { $this lg:viaRule lg:missingRule . }
    }
  
```

### lg:augmentedRuleCategoricalConsistencyRendering
- `wrapsQueryAST`: `lg:augmentedRuleCategoricalConsistencyQuery`

```sparql
    SELECT $this
    WHERE {
      $this lg:hasTypingClause ?t ;
            lg:hasCategoricalClause ?c .
      FILTER NOT EXISTS { ?c lg:indexedByTypingClause ?t . }
    }
  
```

### lg:augmentedRuleTypingConsistencyRendering
- `wrapsQueryAST`: `lg:augmentedRuleTypingConsistencyQuery`

```sparql
    SELECT $this
    WHERE {
      $this lg:hasSyntaxClause ?s ;
            lg:hasTypingClause ?t .
      FILTER NOT EXISTS { ?t lg:fiberOverSyntaxClause ?s . }
    }
  
```

### lg:evolutionIdentifierPersistenceRendering
- `wrapsQueryAST`: `lg:evolutionIdentifierPersistenceQuery`

```sparql
    SELECT $this
    WHERE {
      $this lg:viaRule ?r ;
            lg:toGrammarState ?g2 .
      ?r lg:introducesIdentifier ?i .
      FILTER NOT EXISTS { ?g2 lg:hasEmergentIdentifier ?i . }
    }
  
```

### lg:evolutionRulePersistenceRendering
- `wrapsQueryAST`: `lg:evolutionRulePersistenceQuery`

```sparql
    SELECT $this
    WHERE {
      $this lg:viaRule ?r ;
            lg:toGrammarState ?g2 .
      FILTER NOT EXISTS { ?g2 lg:hasRule ?r . }
    }
  
```

### lg:queryRenderingMatchesSelectTextCheckRendering
- `wrapsQueryAST`: `lg:queryRenderingMatchesSelectTextCheck`

```sparql
    SELECT $this
    WHERE {
      $this lg:hasQueryRendering ?r ;
            sh:select ?s .
      FILTER NOT EXISTS { ?r rdf:value ?s . }
    }
  
```

### lg:queryRenderingWrapsSameASTCheckRendering
- `wrapsQueryAST`: `lg:queryRenderingWrapsSameASTCheck`

```sparql
    SELECT $this
    WHERE {
      $this lg:hasQueryAST ?q ;
            lg:hasQueryRendering ?r .
      FILTER NOT EXISTS { ?r lg:wrapsQueryAST ?q . }
    }
  
```

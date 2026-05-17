import great_expectations as gx
from typing import Tuple, List

"""
This function is the last function in the pipeline

The feature engineered data has been passed to this function

The main goal of this function is to 
    Validate the date using Great Expectations
    Implement data quality checks that must pass before model training
"""

def validate_data(df) -> Tuple[bool, List[str]]:
    
    print("Starting data validation with Great Expectations...")
    
    #Convert pandas DataFrame to Great Expectations Dataset
    context = gx.get_context()
    assert type(context).__name__ == "EphemeralDataContext"

    # Add pandas datasource
    datasource = context.data_sources.add_pandas(name="data_source")

    # Add dataframe asset
    data_asset = datasource.add_dataframe_asset(name="data_asset")
    
    # Add the Batch Definition
    batch_definition_name = "data_asset_batch"
    batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)
    assert batch_definition.name == batch_definition_name
    
    # Define the Batch Parameters
    batch_parameters = {"dataframe": df}
    # Retrieve the Batch
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    # Create validator
    # Create an Expectation Suite
    expectation_suite_name = "data_asset_suite"
    suite = gx.ExpectationSuite(name = expectation_suite_name)
    
    # Add Expectations
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="inventory_id")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="part_num")
    )
    
    # Add the Expectation Suite to the Context
    ge_df = context.suites.add(suite)
    
    print("Validating schema and required columns...")
    
    #customer_id must exist
    ge_df.expect_column_to_exist("customer_id")
    ge_df.expect_column_values_to_not_be_null("customer_id")
    
    #demographic features
    ge_df.expect_column_to_exist("age") 
    ge_df.expect_column_to_exist("gender") 
    ge_df.expect_column_to_exist("annual_income")
    ge_df.expect_column_to_exist("education")
    ge_df.expect_column_to_exist("marital_status")
    ge_df.expect_column_to_exist("dependents")
    
    #service feature
    ge_df.expect_column_to_exist("contract")
    ge_df.expect_column_to_exist("customer_satisfaction")
    ge_df.expect_column_to_exist("num_complaints")
    ge_df.expect_column_to_exist("avg_monthly_gb")
    
    #financial featuere
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("monthlycharges")
    ge_df.expect_column_to_exist("totalcharges")
    ge_df.expect_column_to_exist("late_payments")
    ge_df.expect_column_to_exist("credit_score")
    
    print("Validating business logic constraints...")
    
    #ensuring data integrity
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female", "Other"])
    
    #validate Yes/No fields
    ge_df.expect_column_values_to_be_in_set("paperless_billing", ["Yes", "No"])

    #validate contract types
    ge_df.expect_column_values_to_be_in_set(
        "contract", 
        ["month-to-month", "one_year", "two_year"]
    )
    
    #validate payment method
    ge_df.expect_column_values_to_be_in_set(
        "payment_method",
        ['electronic_check', 'bank_transfer', 'credit_card', 'mailed_check']
    )

    print("Validating numeric ranges and business constraints...")
    
    #validating monthly charges (nonnegative)
    ge_df.expect_column_values_to_be_between("monthlycharges", min_value=0)
    
    #validating monthly charges (nonnegative)
    ge_df.expect_column_values_to_be_between("totalcharges", min_value=0)
    
    print("Validating statistical properties...")
    
    #validating tenure (nonnegative and below 120 months)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    
    #validating tenure (below RM900)
    ge_df.expect_column_values_to_be_between("monthlycharges", min_value=0, max_value=900)
    
    #should be no missing values
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("monthlycharges")
    
    print("Validating data consistency...")
    
    #validating total charges should generally be >= monthly charges
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="totalcharges",
        column_B="monthlycharges",
        or_equal=True,
        mostly=0.95  
    )
    
    print("Running complete validation suite...")
    results = ge_df.validate()
    
    #extract failed expectation into error reporting
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)
    
    #validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return results["success"], failed_expectations
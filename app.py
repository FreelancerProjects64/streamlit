import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyarrow import feather
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import matplotlib.dates as mdates
from components.Risk_Diff.cart_model import CARTModel


class FinancialAnalysisApp:
    def __init__(self):
        self.title = "Financial Analysis Application"
        self.data_folder = "ImportedData"

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        if "data" not in st.session_state:
            st.session_state.data = None
        if 'cf_data' not in st.session_state:
            st.session_state.cf_data = None

    def run(self):
        # st.markdown(f"<h1 style='text-align: center;'>{self.title}</h1>", unsafe_allow_html=True)
        self.show_navbar()

    def show_navbar(self):
        tabs = st.tabs(["File", "Data Analysis", "Risk Diff", "Maximum Recovery", "CF Projection",
                        "Risk Quant", "Predictive Ability"])

        with tabs[0]:
            self.file_menu()
        with tabs[1]:
            self.data_analysis_menu()
        with tabs[2]:
            self.risk_diff_menu()
        with tabs[3]:
            self.maximum_recovery_menu()
        with tabs[4]:
            self.cf_projection_menu()
        with tabs[5]:
            self.risk_quant_menu()
        with tabs[6]:
            self.predictive_ability_menu()

    def file_menu(self):
        tabs = st.tabs(["Import", "Export"])
        if tabs[0]:
            self.import_data()
        elif tabs[1]:
            self.export_data()

    def data_analysis_menu(self):
        sub_tabs = st.tabs(["Data Quality", "Extreme Values", "Discretization"])
        with sub_tabs[0]:
            self.data_quality_analysis()
        with sub_tabs[1]:
            st.write("Extreme value detection and analysis.")
        with sub_tabs[2]:
            st.write("Discretization methods and techniques.")

    def risk_diff_menu(self):
        sub_tabs = st.tabs(["CART", "Discrimination Power", "Heterogeneity", "Homogeneity", "Time Stability"])
        with sub_tabs[0]:
            cart_tabs = st.tabs(["Sampling", "Cart", "Model Analysis"])
            with cart_tabs[0]:
                st.write("CART sampling analysis.")
            with cart_tabs[1]:
                self.cart_sub_menu()
            with cart_tabs[2]:
                st.write("CART model analysis.")
        with sub_tabs[1]:
            dp_tabs = st.tabs(["GAUC", "MAE"])
            with dp_tabs[0]:
                st.write("Global Area Under the Curve (GAUC) analysis.")
            with dp_tabs[1]:
                st.write("Mean Absolute Error (MAE) analysis.")

    def extreme_values_analysis(self):
        st.write("Extreme value detection and analysis under development.")

    def maximum_recovery_menu(self):
        tabs = st.tabs(["CF Analysis", "Number Analysis"])
        if tabs[0]:
            self.mx_recovery_cf_analysis()
        elif tabs[1]:
            st.write("Numerical data analysis.")

    def cf_projection_menu(self):
        tabs = st.tabs(["Additive Model", "Multiplicative Model", "Uncertainty MoC"])
        if tabs[0]:
            st.write("Additive model cash flow projection.")
        elif tabs[1]:
            st.write("Multiplicative model cash flow projection.")
        elif tabs[2]:
            st.write("Uncertainty Margin of Conservatism (MoC) model.")

    def risk_quant_menu(self):
        tabs = st.tabs(["LRA LGD", "MoC C", "MoC Downturn"])
        if tabs[2]:
            sub_tab = st.tabs(["Dwt Identification", "Dwt Impact"])
            if sub_tab[0]:
                st.write("Downturn Weighting Identification.")
            elif sub_tab[1]:
                st.write("Impact assessment under downturn conditions.")

    def predictive_ability_menu(self):
        tabs = st.tabs(["Backtesting"])
        if tabs[0]:
            st.write("Backtesting model predictions.")

    def import_data(self):
        st.write("Functionality to import data.")
        uploaded_file = st.file_uploader("Choose a CSV, Excel, or Rds file", type=["csv", "xlsx", "rds"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    st.session_state.data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.rds'):
                    st.session_state.data = feather.read_feather(uploaded_file)
                st.write("Data imported successfully!")
                if st.session_state.data is not None and not st.session_state.data.empty:
                    st.dataframe(st.session_state.data)
                else:
                    st.warning("Imported data is empty.")
            except Exception as e:
                st.error(f"Error importing data: {str(e)}")

        # Display data if it exists in session state
        if "data" in st.session_state and st.session_state.data is not None:
            st.dataframe(st.session_state.data)
            if st.button("Save Data"):
                self.save_data(uploaded_file.name)

    def load_data(self, file):
        data = feather.read_feather(file)
        data['Default_Date'] = pd.to_datetime(data['Default_Date'])
        data['year_def'] = data['Default_Date'].dt.year
        data['Reference_Date'] = pd.to_datetime(data['Reference_Date'])
        data['year_ref'] = data['Reference_Date'].dt.year
        data['TiD_ymo'] = data['year_ref'] - data['year_def']
        data['Default_ID'] = data['Facility_ID'].astype(str) + '_' + data['Default_Date'].astype(str)
        data = data.sort_values(by=['Default_ID', 'Reference_Date'])
        data['CF_rate'] = np.divide(data['Discounted_CF'], data['Balance_at_def'])
        data['CF_rate_cumulative'] = data.groupby(['Default_ID'])['CF_rate'].cumsum()
        LGD_ID_last = data.drop_duplicates(subset='Default_ID', keep='last')
        return LGD_ID_last

    def save_data(self, original_filename):
        csv_filename = os.path.join(self.data_folder, original_filename)
        st.session_state.data.to_csv(csv_filename, index=False)
        # Allow users to download the data as CSV
        csv = st.session_state.data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name=original_filename, mime='text/csv')
        st.success(f"Data saved to {csv_filename}")

    def export_data(self):
        st.write("Functionality to export data.")
        st.download_button("Download CSV", data="Sample data here", file_name="export.csv")

    def data_quality_analysis(self):
        if st.session_state.data is not None:
            data = st.session_state.data
            if not data.empty:
                st.subheader("Data Quality Analysis")
                missing_values = data.isnull().sum()
                st.subheader("Missing Values")
                st.write(missing_values[missing_values > 0])
                st.subheader("Outlier Detection (z-score method)")
                outliers = (np.abs(zscore(data.select_dtypes(include=[np.number]))) > 3).sum(axis=0)
                st.write(outliers[outliers > 0])
            else:
                st.warning("The imported data is empty. Please import a valid dataset.")
        else:
            st.warning("No data found. Please import data first.")

    def cart_sub_menu(self):
        st.write("CART model execution.")
        st.write("You can import previously saved data here.")

        # Import data from saved files
        if st.button("Import Data"):
            saved_files = os.listdir(self.data_folder)
            if saved_files:
                selected_file = st.selectbox("Select a file to import", saved_files)
                st.session_state.data = pd.read_csv(os.path.join(self.data_folder, selected_file))
                st.write("Data imported successfully!")
            else:
                st.warning("No saved files found.")

        # Display data if it exists in session state
        if st.session_state.data is not None and not st.session_state.data.empty:
            st.dataframe(st.session_state.data)
            st.sidebar.header("Decision Tree Parameters")
            params = {
                'criterion': st.sidebar.selectbox("Criterion",
                                                  ["squared_error", "friedman_mse", "absolute_error", "poisson"]),
                'splitter': st.sidebar.selectbox("Splitter", ["best", "random"]),
                'max_depth': st.sidebar.slider("Max Depth", 1, 20, 5),
                'min_samples_split': st.sidebar.slider("Min Samples Split", 1, 100, 2) / 100,
                'min_samples_leaf': st.sidebar.slider("Min Samples Leaf", 1, 10, 1),
                'min_weight_fraction_leaf': st.sidebar.slider("Min Weight Fraction Leaf", 0.0, 0.5, 0.0),
                'max_features': st.sidebar.slider("Max Features", 1, 10, 5),
                'max_leaf_nodes': st.sidebar.slider("Max Leaf Nodes", 2, 100, 20),
                'min_impurity_decrease': st.sidebar.slider("Min Impurity Decrease", 0.0, 1.0, 0.0),
                'ccp_alpha': st.sidebar.slider("CCP Alpha", 0.0, 0.5, 0.0)
            }

            self.compute_cart_model(params)
        else:
            st.warning("No data found. Please import data first.")

    def compute_cart_model(self, params):
        if "data" in st.session_state:
            data = st.session_state.data
            train_data, test_data = train_test_split(data, test_size=0.3, random_state=1)
            train_data['Grade'] = pd.Categorical(train_data['Grade'])
            train_data['Calibration_Segment'] = pd.Categorical(train_data['Calibration_Segment'])
            test_data['Grade'] = pd.Categorical(test_data['Grade'], categories=train_data['Grade'].cat.categories)
            test_data['Calibration_Segment'] = pd.Categorical(test_data['Calibration_Segment'], categories=train_data[
                'Calibration_Segment'].cat.categories)

            predictors = ['Collateral_valuation', 'WRO_amt', 'Grade_encoded', 'Calibration_Segment_encoded']
            train_data['Grade_encoded'] = train_data['Grade'].cat.codes
            train_data['Calibration_Segment_encoded'] = train_data['Calibration_Segment'].cat.codes
            test_data['Grade_encoded'] = test_data['Grade'].cat.codes
            test_data['Calibration_Segment_encoded'] = test_data['Calibration_Segment'].cat.codes

            model = CARTModel(**params)
            model.train(train_data[predictors], train_data['LGD_Realised'])
            train_predictions = model.predict(train_data[predictors])
            test_predictions = model.predict(test_data[predictors])
            train_rmse, train_r2 = model.calculate_metrics(train_data['LGD_Realised'], train_predictions)
            test_rmse, test_r2 = model.calculate_metrics(test_data['LGD_Realised'], test_predictions)

            train_metrics = {'RMSE': train_rmse, 'R2': train_r2, 'gAUC': f"{self.gAUC_function(train_data) * 100:.2f}%"}
            test_metrics = {'RMSE': test_rmse, 'R2': test_r2, 'gAUC': f"{self.gAUC_function(test_data) * 100:.2f}%"}
            dis_metric_col1, dis_metric_col2 = st.columns(2)
            with dis_metric_col1:
                self.display_metrics(train_metrics, test_metrics)
                # self.show_graphics1(st.session_state.data)
            with dis_metric_col2:
                self.display_feature_important(model, predictors)
                # self.show_graphics2(st.session_state.data)
            self.display_decision_tree_plot(model, predictors)
            self.display_decision_tree(model, predictors)
            train_data['leaf_node'] = model.regressor.apply(train_data[predictors])
            test_data['leaf_node'] = model.regressor.apply(test_data[predictors])
            self.display_average_lgd_by_year(train_data, "Training")
            self.display_average_lgd_by_year(test_data, "Testing")

    def gAUC_function(self, df):
        return np.random.random()

    def display_metrics(self, train_metrics, test_metrics):
        st.write("### Performance Metrics")
        metrics = pd.DataFrame({
            'Metric': ['gAUC', 'RMSE', 'R-squared'],
            'Training Sample': [train_metrics['gAUC'], train_metrics['RMSE'], train_metrics['R2']],
            'Testing Sample': [test_metrics['gAUC'], test_metrics['RMSE'], test_metrics['R2']]
        })
        st.write(metrics)

    def display_feature_important(self, model, predictors):
        st.write("### Feature Importance's")
        st.write(model.feature_importances(predictors))

    def display_decision_tree_plot(self, model, predictors):
        st.write("### Decision Tree Plot")
        model.plot_tree_structure(predictors)

    def display_decision_tree(self, model, predictors):
        st.write("### Tree Structure")
        st.text(model.export_tree_text(predictors))

    def show_graphics1(self, data):
        if "LGD_Predicted" in data.columns and "LGD_Realised" in data.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(data["LGD_Realised"], data["LGD_Predicted"], alpha=0.5, color='blue',
                       label="Predicted vs Actual")
            ax.plot([data["LGD_Realised"].min(), data["LGD_Realised"].max()],
                    [data["LGD_Realised"].min(), data["LGD_Realised"].max()],
                    color='red', linestyle='--', linewidth=2, label="Perfect Prediction")

            ax.set_xlabel("Actual LGD")
            ax.set_ylabel("Predicted LGD")
            ax.legend()
            ax.set_title("Model Prediction Performance")
            st.pyplot(fig)

            residuals = data["LGD_Realised"] - data["LGD_Predicted"]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(residuals, bins=30, color='purple', alpha=0.7)
            ax.set_xlabel("Residuals")
            ax.set_ylabel("Frequency")
            ax.set_title("Residual Distribution")
            st.pyplot(fig)

    def show_graphics2(self, data):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].hist(data["Collateral_valuation"], bins=30, color='green', alpha=0.7)
        axs[0].set_xlabel("Collateral Valuation")
        axs[0].set_ylabel("Frequency")
        axs[0].set_title("Collateral Valuation Distribution")

        axs[1].hist(data["Grade_encoded"], bins=10, color='orange', alpha=0.7)
        axs[1].set_xlabel("Grade (Encoded)")
        axs[1].set_ylabel("Frequency")
        axs[1].set_title("Grade Distribution")
        plt.tight_layout()
        st.pyplot(fig)

    def display_average_lgd_by_year(self, data, label):
        st.write(f"### Average LGD by Year - {label}")
        avg_lgd_by_year_leaf = data.groupby(['leaf_node', 'year_def'])['LGD_Realised'].mean().unstack()
        fig, ax = plt.subplots(figsize=(12, 6))
        avg_lgd_by_year_leaf.T.plot(ax=ax, marker='o', linewidth=1.5)

        # Formatting for improved readability
        ax.set_xlabel("Year of Default")
        ax.set_ylabel("Average LGD Realised")
        ax.set_title(f"Average LGD by Year and Leaf Node - {label}")
        ax.legend(title="Leaf Node")
        plt.xticks(rotation=45)

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        st.pyplot(fig)

    def mx_recovery_cf_analysis(self):
        st.subheader("Cash Flow Analysis")
        if st.button("Load CF Data"):
            saved_files = os.listdir(self.data_folder)
            if saved_files:
                selected_file = st.selectbox("Select a file to import", saved_files)
                st.session_state.cf_data = pd.read_csv(os.path.join(self.data_folder, selected_file))
                st.write("Data imported successfully!")
            else:
                st.warning("No saved files found.")

        # Show CF data if available
        if st.session_state.cf_data is not None:
            st.write("Cash Flow Data Preview:")
            st.dataframe(st.session_state.cf_data.head())

            data = st.session_state.cf_data
            LGD_ID_closed = data.query("Closed_Scenario_ID != 'DEF'")
            Triangle = LGD_ID_closed.pivot_table(values='marginal_rec_rate_cap', index='year_def', columns='TiD_ymo', aggfunc='mean')

            # Calculate cumulative and incremental cash flows
            column_means = Triangle.mean()
            cumulative_sum = column_means.cumsum()
            cumulative_sum_inversed = column_means[::-1].cumsum()[::-1]

            # Locate the MRP threshold position
            threshold = 0.05
            MRP_pos = next((len(cumulative_sum_inversed) - 1 - i for i in range(len(cumulative_sum_inversed) - 1, -1, -1)
                            if cumulative_sum_inversed[i] > threshold), None)

            # Display MRP position in UI
            if MRP_pos is not None:
                st.info(f"MRP Position: {MRP_pos}")

            # Plot the cumulative cash flow and threshold line
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(cumulative_sum, marker='o', linestyle='-', color='b', label='Cumulative Recoveries')
            if MRP_pos is not None:
                ax.axvline(x=MRP_pos, color='r', linestyle='--', label='MRP Threshold')
            ax.set_xlabel("Index")
            ax.set_ylabel("Cumulative Recovery")
            ax.set_title("Cash Flow Cumulative Analysis")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No data loaded. Please load CF data to proceed.")

# Run the application
if __name__ == "__main__":
    app = FinancialAnalysisApp()
    app.run()

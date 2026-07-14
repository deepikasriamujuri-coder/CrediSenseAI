/* --------------------------------------------------
   CrediSense AI - Client-side Interactions
   -------------------------------------------------- */

document.addEventListener("DOMContentLoaded", function() {
    
    // 1. Inputs Formatting & Real-Time Ratio Calculations
    const incomeInput = document.getElementById("person_income");
    const loanInput = document.getElementById("loan_amnt");
    const ratioInput = document.getElementById("loan_percent_income");
    
    // Create or locate helper texts
    let incomeHelper = document.createElement("small");
    incomeHelper.style.color = "#2a9d8f";
    incomeHelper.style.display = "block";
    incomeHelper.style.marginTop = "3px";
    if (incomeInput) incomeInput.parentNode.appendChild(incomeHelper);
    
    let loanHelper = document.createElement("small");
    loanHelper.style.color = "#2a9d8f";
    loanHelper.style.display = "block";
    loanHelper.style.marginTop = "3px";
    if (loanInput) loanInput.parentNode.appendChild(loanHelper);
    
    function formatCurrency(val) {
        if (!val || isNaN(val)) return "";
        return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);
    }
    
    function updateCalculations() {
        const income = parseFloat(incomeInput ? incomeInput.value : 0);
        const loan = parseFloat(loanInput ? loanInput.value : 0);
        
        // Update helper formatting text
        if (income > 0) {
            incomeHelper.textContent = `${formatCurrency(income)} annual income`;
        } else {
            incomeHelper.textContent = "";
        }
        
        if (loan > 0) {
            loanHelper.textContent = `${formatCurrency(loan)} requested loan`;
        } else {
            loanHelper.textContent = "";
        }
        
        // Auto-calculate loan percentage of income
        if (income > 0 && loan > 0 && ratioInput) {
            const ratio = loan / income;
            // Set value up to 4 decimal places
            ratioInput.value = Math.min(ratio, 1.0).toFixed(4);
        }
    }
    
    if (incomeInput) incomeInput.addEventListener("input", updateCalculations);
    if (loanInput) loanInput.addEventListener("input", updateCalculations);
    
    // 2. Form Submission Loading States
    const form = document.querySelector("form");
    if (form) {
        form.addEventListener("submit", function(event) {
            const submitBtn = form.querySelector("button[type='submit']");
            if (submitBtn) {
                // Disable to prevent double submission
                submitBtn.disabled = true;
                submitBtn.style.opacity = "0.7";
                submitBtn.style.cursor = "not-allowed";
                
                // Set loading content
                submitBtn.innerHTML = '<span class="spinner"></span> Analyzing Credit Risk...';
            }
        });
    }
    
    // 3. Mobile Navigation Menu Toggle
    const toggleBtn = document.querySelector(".nav-toggle");
    const navLinks = document.querySelector(".nav-links");
    
    if (toggleBtn && navLinks) {
        toggleBtn.addEventListener("click", function() {
            if (navLinks.style.display === "flex") {
                navLinks.style.display = "none";
            } else {
                navLinks.style.display = "flex";
                navLinks.style.flexDirection = "column";
                navLinks.style.position = "absolute";
                navLinks.style.top = "70px";
                navLinks.style.right = "0";
                navLinks.style.left = "0";
                navLinks.style.backgroundColor = "var(--bg-secondary)";
                navLinks.style.padding = "20px";
                navLinks.style.borderBottom = "1px solid var(--border-color)";
                navLinks.style.gap = "15px";
                navLinks.style.alignItems = "center";
            }
        });
    }
});

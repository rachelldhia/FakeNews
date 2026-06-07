document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const newsInput = document.getElementById('newsInput');
    const charCount = document.getElementById('charCount');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    
    const feedbackArea = document.getElementById('feedbackArea');
    const loadingState = document.getElementById('loadingState');
    const loadingText = document.getElementById('loadingText');
    const resultState = document.getElementById('resultState');
    const errorState = document.getElementById('errorState');
    const errorText = document.getElementById('errorText');
    
    const verdictBadge = document.getElementById('verdictBadge');
    const realPct = document.getElementById('realPct');
    const fakePct = document.getElementById('fakePct');
    const realProgressBar = document.getElementById('realProgressBar');
    const fakeProgressBar = document.getElementById('fakeProgressBar');
    const insightsList = document.getElementById('insightsList');
    
    const mobileNavToggle = document.querySelector('.mobile-nav-toggle');
    const navMenu = document.getElementById('navMenu');

    // --- Mobile Menu Toggle ---
    if (mobileNavToggle && navMenu) {
        mobileNavToggle.addEventListener('click', () => {
            mobileNavToggle.classList.toggle('active');
            navMenu.classList.toggle('active');
        });

        // Close mobile menu when clicking a link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                mobileNavToggle.classList.remove('active');
                navMenu.classList.remove('active');
            });
        });
    }

    // --- Active Nav Links on Scroll ---
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (pageYOffset >= (sectionTop - 120)) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            const href = link.getAttribute('href');
            if (href === `#${current}` || (current === '' && href === '#')) {
                link.classList.add('active');
            }
        });
    });

    // --- Character Count Observer ---
    newsInput.addEventListener('input', () => {
        const length = newsInput.value.length;
        charCount.textContent = `${length.toLocaleString()} / 5,000 characters`;

        // Style warnings based on content length
        if (length >= 4900) {
            charCount.className = 'char-count limit';
        } else if (length >= 4000) {
            charCount.className = 'char-count warning';
        } else {
            charCount.className = 'char-count';
        }
    });

    // --- Clear Input ---
    clearBtn.addEventListener('click', () => {
        newsInput.value = '';
        charCount.textContent = '0 / 5,000 characters';
        charCount.className = 'char-count';
        feedbackArea.classList.add('hidden');
        loadingState.classList.add('hidden');
        resultState.classList.add('hidden');
        errorState.classList.add('hidden');
        newsInput.focus();
    });

    // --- Prediction API call ---
    const loadingPhases = [
        "Initializing AI pipeline...",
        "Parsing linguistic features...",
        "Evaluating vocabulary distribution...",
        "Comparing with verified datasets...",
        "Calculating prediction confidence..."
    ];

    analyzeBtn.addEventListener('click', async () => {
        const text = newsInput.value.trim();

        if (!text) {
            alert('Please paste some news article content to analyze.');
            newsInput.focus();
            return;
        }

        // 1. Prepare UI for loading
        feedbackArea.classList.remove('hidden');
        loadingState.classList.remove('hidden');
        resultState.classList.add('hidden');
        errorState.classList.add('hidden');

        // Disable elements during analysis
        setFormDisabled(true);

        // 2. Start dynamic loading phrases
        let phaseIndex = 0;
        loadingText.textContent = loadingPhases[phaseIndex];
        const phaseInterval = setInterval(() => {
            phaseIndex = (phaseIndex + 1) % loadingPhases.length;
            loadingText.textContent = loadingPhases[phaseIndex];
        }, 1200);

        try {
            // 3. Make HTTP request to the Flask local server
            const response = await fetch('http://localhost:5001/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            clearInterval(phaseInterval);

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Server error occurred during prediction.');
            }

            const data = await response.json();

            // 4. Render prediction results
            showResult(data);

        } catch (err) {
            clearInterval(phaseInterval);
            showError(err.message);
        } finally {
            setFormDisabled(false);
        }
    });

    function setFormDisabled(disabled) {
        newsInput.disabled = disabled;
        analyzeBtn.disabled = disabled;
        clearBtn.disabled = disabled;
        
        if (disabled) {
            analyzeBtn.style.opacity = '0.7';
            analyzeBtn.style.cursor = 'not-allowed';
            clearBtn.style.opacity = '0.5';
            clearBtn.style.cursor = 'not-allowed';
        } else {
            analyzeBtn.style.opacity = '';
            analyzeBtn.style.cursor = '';
            clearBtn.style.opacity = '';
            clearBtn.style.cursor = '';
        }
    }

    function showResult(data) {
        loadingState.classList.add('hidden');
        resultState.classList.remove('hidden');

        const isReal = data.prediction === 'Real News';
        
        // Update verdict badge
        verdictBadge.textContent = data.prediction;
        if (isReal) {
            verdictBadge.className = 'badge real';
        } else {
            verdictBadge.className = 'badge fake';
        }

        // Get percentages
        const realVal = data.confidence.real_probability_percent;
        const fakeVal = data.confidence.fake_probability_percent;

        // Animate percentage texts
        animateValue(realPct, 0, realVal, 1000);
        animateValue(fakePct, 0, fakeVal, 1000);

        // Animate progress bars
        setTimeout(() => {
            realProgressBar.style.width = `${realVal}%`;
            fakeProgressBar.style.width = `${fakeVal}%`;
        }, 100);

        // Generate Insights bullets
        insightsList.innerHTML = '';
        const insights = getInsightsForPrediction(isReal, realVal, fakeVal, newsInput.value.trim());
        insights.forEach(insight => {
            const li = document.createElement('li');
            li.textContent = insight;
            insightsList.appendChild(li);
        });
    }

    function showError(message) {
        loadingState.classList.add('hidden');
        errorState.classList.remove('hidden');
        
        if (message.includes('Failed to fetch')) {
            errorText.textContent = "Could not connect to the Prediction API. Please make sure your Python Flask server is running locally at http://localhost:5001.";
        } else {
            errorText.textContent = message;
        }
    }

    function getInsightsForPrediction(isReal, realVal, fakeVal, text) {
        const wordCount = text.split(/\s+/).length;
        const sentenceCount = text.split(/[.!?]+/).filter(Boolean).length;
        const avgSentenceLength = Math.round(wordCount / (sentenceCount || 1));
        
        const list = [];
        
        if (isReal) {
            list.push(`Linguistic patterns show an analytical, informative, and objective tone (Confidence: ${realVal}%).`);
            list.push(`The structure exhibits consistent vocabulary choices aligned with standard journalism and factual reporting.`);
            if (avgSentenceLength > 18) {
                list.push(`Average sentence length (${avgSentenceLength} words) is typical of complex, comprehensive analytical articles.`);
            } else {
                list.push(`Text layout contains clear, direct statements typical of breaking reports.`);
            }
            list.push(`Absence of clickbait exclamation markers, excessive capitalization, or emotionally loaded keywords.`);
        } else {
            list.push(`High density of sensationalized words, hyper-partisan vocabulary, or strong opinion markers detected (Confidence: ${fakeVal}%).`);
            list.push(`The syntax pattern strongly aligns with typical fabricated or low-credibility digital copy.`);
            if (text.includes('!') || text.includes('?') || text.toUpperCase() === text) {
                list.push(`Contains high-activation punctuation patterns or capitalizations designed to solicit emotional reactions.`);
            }
            if (wordCount < 100) {
                list.push(`The article text is relatively short (${wordCount} words), typical of clickbait headlines lacking structural elaboration.`);
            } else {
                list.push(`Linguistic complexity drops significantly in parts, which is standard in copycat or non-factual writeups.`);
            }
        }
        
        return list;
    }

    // --- Animated counter utility ---
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const current = Math.floor(progress * (end - start) + start);
            obj.innerHTML = `${current}%`;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});

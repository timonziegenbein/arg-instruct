output_first_template_for_clf = '''Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, just generate possible class labels.

Task: Is the following argument clause a premise?
Class label: Yes
Input: The Commission notes that the applicant was detained after having been sentenced by the first instance court to 18 months' imprisonment
Class label: No
Input: It follows that this part of the application must be rejected for non-exhaustion of domestic remedies, in accordance with Article 27 para. 3 (Art. 27-3) of the Convention.

Task: An argument is missing relevance if it does not discuss the issue, but derails the discussion implicitly towards a related issue or shifts completely towards a different issue. Given the following argument and the topic of the debate the argument appeared in. Decide whether the argument Lacks Relevance or Does Not Lack Relevance
Class label: Lacks Relevance
Input: Topic: Firefox vs internet explorer:
Argument: That form of argument degrades this forum, and will cause the arguments to fall to the lowest common denominator.
This word, "indisputable", I do not think it means what you think it does. I dispute your claim from personal experience, if nothing else. That makes it disputable. I think Yahoo! has chat rooms more attuned to your style of debate. Check them out.
Class label: Does Not Lack Relevance
Input: Topic: If your spouse committed murder and he or she confided in you would you turn them in:
Argument: I wouldnt turn her in becuase she is my wife. She made a mistake that we can get over it. If she trusted me by telling me what she did then I couldn't do that to her.

Task: Given the following two arguments (Argument A and Argument B), determine which of the two is more convincing.
Class label: Argument A
Input: Argument A: I am a nurse and the more I studied biology, the more I was in awe of the miracle of our bodies. Every tiny cell has it's own design that is even more complex than a computer. When you look at a computer do you question that there had to be a creator? Or do you need proof that this highly organized machine came to be about by a mere accident or order of circumstances? Evolutionists need more faith than Creationists.
Argument B: k, I believe that 'God' created and then tweaked via evolution? <br/> we all find out within 100 years anyways....
Class label: Argument B
Input: Argument A: i dont think soooo even as an indian citizen.........because of soo much unemployment i dont think the gdp of india will increase in the next 5 years....... they dont even have enough funds to feed all the poor..how can we even imagine them to lead the world...........
Argument B: All we can say is a big IF because we are not yet 100% sure if this nation itself could lead the whole world. Who knows maybe in the future that India could fail to lead us. There are still other countries who are more progressive, more literated, more powerful than India, like the USA, United Kingdom, Russia, Japan, etc. This nation doesn't have yet enought potentials and ability to sustain the needs of the people. India is not yet that full of technologies like in the Americas. Well in fact technology is one of the most sophisticated and essential inventions ever created.

Task: What kind of support relation, if any, exists from elementary unit X for a proposition Y of the same argument? Differentiate between REASON, EVIDENCE and NO SUPPORT RELATION. Support relations in this scheme are two prevalent ways in which propositions are supported in practical argumentation: REASON and EVIDENCE. The former can support either objective or subjective propositions, whereas the latter can only support objective propositions. That is, you cannot prove that a subjective proposition is true with a piece of evidence. REASON: For an elementary unit X to be a REASON for a proposition Y, it must provide a reason or a justification for Y. For example, “The only issue I have is that the volume starts to degrade a little bit after about six“and I find I have to buy a new pair every year or so.”(Y). EVIDENCE: For an elementary unit X to be EVIDENCE for a proposition Y, it must prove that Y is true. For example, “https://images-na.ssl-images-amazon.com/[...]”(X) and “The product arrived damage[d],”(Y).
Class label: REASON
Input: Argument: Excellant muisc reproduction and outside noise reduction. I cannot hear anything but music, even in a gym with loud music and people talking. I own Klipsch speakers for surround sound speakers on TV - awesome. Klipsch makes quality gear.
Elementary unit X: I cannot hear anything but music, even in a gym with loud music and people talking.
Proposition Y: Excellant muisc reproduction and outside noise reduction.
Class label: EVIDENCE
Input: Argument: The East Version from Helix Trade is definitely fake. The package and the product are both totally different from the genuine one my friends got from Germany. And its sound quality is way more worse than the real version. It is just an bad level copy version of IE80.
Elementary unit X: The package and the product are both totally different from the genuine one my friends got from Germany.
Proposition Y: The East Version from Helix Trade is definitely fake.
Class label: NO SUPPORT RELATION
Input: Argument: worst product I ever bought on Amazon. I bought a blue and a pink and this worked only for 24 hours !!!! Absurd tremendous! I want my money back ! quality horrible
Elementary unit X: I bought a blue and a pink
Proposition Y: Absurd tremendous!

Task: Compare the given two versions of the same claim and determine which one is better (Claim 1 or Claim 2).
Class label: Claim 1
Input: Claim 1: It takes years for racism to be organically reduced and for people to have a change of opinions. A radical policy that seeks to speed up this process is too drastic of a change being forced about people.
Claim 2: Racism takes years to organically be reduced and people to have a change of opinions. A radical policy that seeks to speed up this process is too drastic of a change being forced about people.
Class label: Claim 2
Input: Claim 1: It's makes sense to me that marriage should be treated like a business partnership. I believe the basic risks are no different. 
Currently experiencing a divorce, I have no doubts that money, time, and peace of mind would be preserved with an agreement put in place before marriage.
Claim 2: Marriage should be treated like a business partnership with no less. The long-term risks only vary by kind and not degree of impact on the relationship. Money, time, and peace of mind can be preserved with an agreement put in place before marriage.

Task: The edges representing arguments are those that connect argumentative discourse units (ADUs). The scheme distinguishes between supporting and attacking relations. Supporting relations are normal support and support by example. Attacking relations are rebutting attacks (directed against another node, challenging the acceptability of the corresponding claim) and undercutting attacks (directed against another relation, challenging the argumentative inference from the source to the target of the relation). Finally, additional premises of relations with more than one premise are represented by additional source relations. Given the following two argumentative discourse units, determine the function of the segment, i.e. support, support by example, rebutting attack, undercutting attack, or additional premise.
Class label: additional premise
Input: ADU1: The ecosystem needs this delicate balance,
ADU2: In the absence of wolves and other large predators, humans become the deer population control to keep to deer from over-grazing regional flora.
Class label: rebutting attack
Input: ADU1: However, being vegetarian requires more vigilance in terms of sufficient vitamins, which meat eaters need not concern.
ADU2: Overall, being vegetarian is easier on the environment
Class label: support
Input: ADU1: I think that, to some extent, romantic movies can leave an impression on viewers about what they think their real life relationships should be.
ADU2: Yes, the expectations raised by romantic movies are damaging to real relationships.
Class label: support by example
Input: ADU1: It also makes people feel like they are really making an impact on the environment by doing something that is very easy and doesn't take much time to do.
ADU2: There are many benefits to recycling,
Class label: undercutting attack
Input: ADU1: but because there are no limits teens often wind up spending more time in that virtual world than in the real world.
ADU2: It is true that social media is very beneficial for staying in contact with people far away,

Task: Argument conclusions are valid if they follow from the premise, meaning a logical inference links the premise to the conclusion. Given the conclusions below: Is conclusion A better than conclusion B in terms of validity?
Class label: Yes
Input: Premise: Torture puts the torturer in a position of dominance and abuse that has a brutalizing effect. This brutalizing effect is dehumanizing, or at least it defeats the virtues of compassion, empathy, and dignity that define a good human being, perhaps in God's image.
Conclusion A: Torture dehumanizes the torturer
Conclusion B: Trying terrorists risks releasing intelligence, costing lives
Class label: They are equally valid
Input: Premise: Third parties often find it difficult to emerge later, even if they have a decent following. The dominant parties tend to shape electoral rules to the exclusion of smaller parties, and the more dominant parties tend to be the most successful at raising funds.
Conclusion A: Two-party system tends to favor dominant parties.
Conclusion B: Third parties are less likely to emerge later.
Class label: No
Input: Premise: Television has become a temple of mass production, shallow values and stereotypes that have a great influence on modern society. This negative effect spreads with the growing popularity of TV, especially among young people and children. It defiantly changes our society for the worse, trivialising culture and making us all conform to a bland, "Hollywood" model of entertainment in which regional traditions and diversity are lost.
Conclusion A: Television is a form of entertainment.
Conclusion B: Television is a temple of shallow values

Task:'''


output_first_template_for_reg = '''Given the regression task definition and the score range, generate an input that corresponds to the lower bound, upper bound and a score in the middle of the range. If the task doesn't require input, just generate possible scores.

Task: The premises of an argument should be seen as sufficient if, together, they provide enough support to make it rational to draw the argument’s conclusion. If you identify more than one conclusion in the comment, try to adequately weight the sufficiency of the premises for each conclusion when judging about their “aggregate” sufficiency—unless there are particular premises or conclusions that dominate your view of the author’s argumentation. Notice that you may see premises as sufficient even though you do not personally accept all of them, i.e., sufficiency does not presuppose acceptability. How would you rate the sufficiency of the premises of the author’s argument on the scale "1" (Low), "2" (Average) or "3" (High)?
Score: 1
Input: Who are we to judge what is right or wrong? Can we not just let people make decisions and live with the consequences?
Score: 2
Input: Easy access to porn may be the reason there has been an 85 percent decline in rapes over the last 25 years, according to a law professor at Northwestern University.
Score: 3
Input: PE should be compulsory because it keeps us constantly fit and healthy. If you really dislike sports, then you can quit it when you're an adult. But when you're a kid, the best thing for you to do is study, play and exercise. If you prefer to be lazy and lie on the couch all day then you are most likely to get sick and unfit. Besides, PE helps kids be better at teamwork.

Task: How would you rate the overall quality of the author’s argumentation on the scale "1" (Low), "2" (Average) or "3" (High)?
Score: 1
Input: it is true that bottled water is a waste, but bottles can be reused!
Score: 2
Input: Most Americans on average recycle 86-88% of there bottle waters. That is more than half! So stop complaining that we dont recycle because we do! <br/> http://www.treehugger.com/culture/23-percent-of-americans-donatmt-recycle.html
Score: 3
Input: Water bottles, good or bad? Many people believe plastic water bottles to be good. But the truth is water bottles are polluting land and unnecessary. Plastic water bottles should only be used in emergency purposes only. The water in those plastic are only filtered tap water. In an emergency situation like Katrina no one had access to tap water. In a situation like this water bottles are good because it provides the people in need. Other than that water bottles should not be legal because it pollutes the land and big companies get 1000% of the profit.

Task: Argument strength refers to the strength of the argument an essay makes for its thesis. An essay with a high argument strength score presents a strong argument for its thesis and would convince most readers. Score the argument strength of the given argumentative essay using the following scoring range:
1.0 (essay does not make an argument or it is often unclear what the argument is)
1.5
2.0 (essay makes a weak argument for its thesis or sometimes even argues against it)
2.5
3.0 (essay makes a decent argument for its thesis and could convince some readers)
3.5
4.0 (essay makes a strong argument for its thesis and would convince most readers)
Score: 1.0
Input: Beeing the son of a policeman I think I am fit for answering the statement made in the title. It has surely affected my life so far and definetly will in the future .
Ever since I was thrown into this world by my mother I have been taught that one should always follow the law. Why? Because your father is a cop and you wouldn't want to embarrass him now would you? Now don't get me wrong here. I understand completely that without some limitations and rules the society would be a rather chaotic place, but one should be allowed to manipulate or twist the law just a little bit sometimes. But with me beeing Mr. Law junior I never have had the chance to do that. I may seem a little bitter about that fact and I guess I am. One of my earliest encounters with the devlish art of law breaking was during my first year at school. Of course this wasn't me doing the breaking but one of classmates or should I say inmates. The classical theft of candy was my first chance to see a criminal in action. And yes you could have guessed it, he got caught. I almost have to laugh when I think of it now. He was by god the worst thief I have ever seen. The first rule of stealing is to make sure that you don't have anybody watching you at the very second you make your move, so to speak.. He did not and therefore he got caught by the baddest 120 kilo grocery store manager you have ever seen. The mean manager pulled him crying into his office and yelled at him so damn loud that to this very day not a singel student at my old school has stolen as much as a paper bag while he was on the job. But this incident or should I call it accident got my little head thinking. Even though he got caught I couldn't help thinking of how easy it must be to steal if you just use your head and plan in advance what you are going to snatch. This was the start of a brilliant career. I started off easy taking only candy and your ordinary groceries like condoms and serial boxes. The thing was that after I had gotten out of the store and not getting caught I always went back in and put whatever I stole back on its shelf. I didn't get busted once. Of course this was only the beginning of a luxurious career. One day I got to the point where groceries had lost its charm and I had to move on to bigger and better things. And what could be more rewarding than stealing cars? The only problem now was that I really couldn't put the cars back after I had taken them. At first I felt kind of bad about this but once you steal your first car and drive off into the sunset the rush is so intense thatyou really don't think of the consequences. I got to know a couple of jail-house regulars who could take the cars off my hands, that means that they buy them from me so you law obeying morons understand what I mean. This gave me a rather solid income and you could say that my every day life changed for the better. Its not usual for a 16 year old kid to walk around being able to buy whatever he feels like. People always asked me how I could buy all the stuff I did but I just said that the police salary is not as bad as one might think, and most people believed me. You may ask if I had lost my mind doing all these things. Of course I had some bad days where I nearly got caught and felt rather strange afterwards. Imagine being arrested by your own father. That has got to be one of the most humiliating experieces one can have. But to this very day I haven't been arrested so the only worry I have is what to spend my money on. Crime does not pay? Huh. It has paid for my entire appartment and education. I have wooed so many girls by taking them to the most fancy places just because I ripped off a car the day before so don't come here and lecture me about moral. I just hope I don't get caught by my old man. That's all.....
Score: 4.0
Input: Money is definitely the root of all evil. I believe that at least 96 % of all the people in the world are not satisfied with the amount of money that they possess. And because earning more money by honest work is too slow and the amount of money you get that way is too small, more and more people resort to dishonest action. But this is only one half of the story, for not only does the yearning for more money cause much physical harm, it also brings along great emotional pain .
Perhaps the most harmful thing about money is the fact that you can never have it enough. No matter how rich you already are, you will always want more, always. You can never just relax and enjoy your money. How sad and torturing it is to live your whole life always trying to reach the state where you no more have to worry about money, but never reaching it .
If you happen to be rich, you probably have one worry more than the poor ones; in the matter of friendship. How can you tell, who is your real friend and who is just some greedy slimeball after your money. As those slimeballs are usually excellent "actors", the only way to reveal their true colours would be to loose all your money. But because that is out of the question, you are stuck with uncertainty. So there is no one around you, who you could, without any doubt, trust .
The lust for money may in some cases also lead to passion for gambling. Las Vegas is the well-known paradise for gamblers all over the world. Many people have lost fortunes there, by the roulette tables. Some people even become addicted to gambling. They simply cannot stop gambling, even though they are up to their ears in debt because of it .
What about the drug business then. The drug lords have realized that you can get huge, gigantic profit in the drug business. And when the stakes are this high, all actions, including brutal murders, are "justified". The most horrible thing is that even some small kids are involved in this business. They are the best possible drug dealers, because no one usually suspects them, and because they cannot be arrested and imprisoned in case they get caught .
Too much money is definitely a bad thing especially for the young. They will never learn to work hard for something they want, if they can always get everything they want with money. For you can even "buy" someone to do your homework and your tests with enough money. This in turn can lead the young one to assume that everything is for sale, even friendship and love .
In some countries, even some policemen and judges are corrupted. With a few, carefully selected bribes, you can buy yourself out of trouble. So, money can even get in the way of justice .
The power of money scares me. I do not want it to get a hold of me. That is why I have decided to aim for managing with as small amount of money as possible .

Task: An argumentation should be seen as reasonable if it contributes to the resolution of the given issue in a sufficient way that is acceptable to everyone from the expected target audience. Try to adequately weight your judgments about global acceptability, global relevance, and global sufficiency when judging about reasonableness—unless there is a particular dimension among these that dominates your view of the author’s argumentation. In doubt, give more credit to global acceptability and global relevance than to global sufficiency due to the limited feasibility of the latter. How would you rate the reasonableness of the author’s argumentation on the scale "1" (Low), "2" (Average) or "3" (High)?
Score: 1
Input: i thik thier bad because i think ushould be free with out nobody telling u wat to do
Score: 2
Input: Water bottles can easily be made into long term fiber materials, like clothing or carpet. It is easy to just fill cup with water and re use it.
Score: 3
Input: Another reason argued against school uniforms is that they deprive the children of their individuality. The stress on a uniform dress code in school opposes the spirit of unity in diversity and its celebration. It is even claimed to restrict socialization, a vital aspect of human nature.

Task: An argument should be seen as cogent if it has individually acceptable premises that are relevant to the argument’s conclusion and that are sufficient to draw the conclusion. Try to adequately weight your judgments about local acceptability, local relevance, and local sufficiency when judging about cogency—unless there is a particular dimension among these that dominates your view of an argument. Accordingly, if you identify more than one argument, try to adequately weight the cogency of each argument when judging about their “aggregate” cogency—unless there is a particular argument that dominates your view of the author’s argumentation. How would you rate the cogency of the author’s argument on the scale "1" (Low), "2" (Average) or "3" (High)?
Score: 1
Input: it is true that bottled water is a waste, but bottles can be reused!
Score: 2
Input: I find that whatever works for a person individually, is good. How would I be able to say something is the best, when, it doesn't work on someone elses computer? I vote for Firefox, because in my experience it has worked multiple times better than Internet Explorer. The config isn't too bad, and those of you who know about:config know what I'm talking about. The tabs are great, and has many good extensions. Though not the best, it is definately superior to Internet Explorer. Especially after the laughable 7 update. ;D
Score: 3
Input: Nope. I believe it shouldnt be done just to discipline a child. Parents could just scold their child just so they would stop what they are doing wrong. Hitting is not really a solution because aside hurting a child emotionally, it also hurts them physically. Kids wouldnt learn the value of respect for their parents because they would only learn how to be scared. They wont do the wrong action again next time, not because they've learned their lesson, but because they are just scared that their parents would hit them

Task: Score the helpfulness of the following review on a scale from 0 (lowest) to 4 (highest).
Score: 0
Input: From the first time I wore them and subsequent times after that I was really amazed how clear and nice they sounded! I've seen them rated in the top 10 on some sites for best earbuds under $50 and that's why I ended up buying them!
Score: 2
Input: For the price you are paying , and the easy set up process this hardware was worthy of me writing a review. I work for a telecom company and have meetings daily. I tested this equipment on my Apple iPhone 6 Plus. I tested the earphone on 2 meetings, and the results were quite great. The range wasn't bad as well approx. 6 meters away from the phone. I would recommend this for casual and simple use.
Score: 4
Input: While the sound is great, the earphone piece snapped off TWICE from normal use within the course of a year. The first time this happened Creative replaced the headphones (though I had to fork over shipping, ~$16) and the second time they offered to fix it for ~$60, which is more than the headphones cost! Complete rip-off and a sleazy way of doing business. I will never purchase another pair of headphones from Creative again.

Task: An argumentation should be seen as successful in creating credibility if it conveys arguments and other information in a way that makes the author worthy of credence, e.g., by indicating the honesty of the author or by revealing the author’s knowledge or expertise regarding the discussed issue. It should be seen as not successful if rather the opposite holds. Decide in dubio pro reo, i.e., if you have no doubt about the author’s credibility, then do not judge him or her to be not credible. How would you rate the success of the author’s argumentation in creating credibility on the scale "1" (Low), "2" (Average) or "3" (High)?
Score: 1
Input: No porn is not wrong. As an abstract noun, it cannot be wrong.
Score: 2
Input: Who are we to judge what is right or wrong? Can we not just let people make decisions and live with the consequences?
Score: 3
Input: As an ambitious, young person wanting to become a lawful, successful, homicide detective, I would not be lenient with any murderer in my midst. <br/> Hopefully, the murder wouldn't be the result of a pleasure/malicious-kill, so that the sentencing won't be as harsh, but nonetheless, all murderers must be tried. After all, hopefully my spouse will understand that having to live in hiding is basically the same as being in prison except much worse since there would be little chance for parole since they will have to live with the guilt and/or the fear of being caught for the rest of their lives.

Task:'''


input_first_template_for_gen = '''Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.

Task: Given as input an argumentative claim, potentially along with context information on the debate, rewrite the claim such that it improves in terms of text quality and/or argument quality, and preserves the meaning as far as possible.
Input: Ending gender inequality is unlikely to occur because of more favourable divorce outcomes for women.
Output: Favourable divorce outcomes for women are unlikely to significantly affect gender inequality.
Input: Contracts for things like signing a lease or starting a new career often have end dates that mean that individuals signing them have a chance of never experiencing the repercussions of ending that contract as they merely exit the contract at the end date. Couples with a prenup who end a marriage by divorce always face the repercussions of having signed that contract.
Output: Leases and career contracts have end dates enshrined in the contract themselves, meaning that a person is not bound by that contract indefinitely. Couples with a prenup who end a marriage by divorce will always face the repercussions of having signed that contract.
Input: The ketubah is only a one-way contract that details what is expected of a husband towards his wife under Halakha (the Jewish Law).
Output: The ketubah is only a one-way contract that details what is expected of a husband towards his wife under Halakha (the Jewish Law) - clothing, food and sex.

Task: Label each elementary argumentative unit as REFERENCE or as one of the proposition types FACT, TESTIMONY, POLICY, and VALUE. FACT (Proposition of Non-Experiential Fact) is an objective proposition, meaning it does not leave any room for subjective interpretations or judgements. For example, “and battery life is about 8-10 hours.”. TESTIMONY (Proposition of Experiential Fact) is also an objective proposition. However, it differs from FACT in that it is experiential, i.e., it describes a personal state or experience. For example, “I own Sennheisers, Bose, Ludacris Souls, Beats, etc.”. POLICY (Proposition of Policy) is a subjective proposition that insists on a specific course of action. For example, “They need to take this product off the market until the issue is resolved.”. VALUE (Proposition of Value) is a subjective proposition that is not POLICY. It is a personal opinion or expression of feeling. For example, “They just weren’t appealing to me”. REFERENCE (Reference to a Resource) is the only non-proposition elementary unit that refers to a resource containing objective evidence. In product reviews, REFERENCE is usually a URL to another product page, image or video. Also, REFERENCE cannot be supported by other elementary units. For example, “https://images-na.ssl-images-amazon.com/[...]”.
Input: Purchased a pair.
Sounded ok.
The jack for the charger broke in less than six months
and they would not give me an exchange.
One of the earpads came off
and you can't replace that.
Shoddy workmanship.
Would not recommend.
Output: TESTIMONY
VALUE
TESTIMONY
TESTIMONY
TESTIMONY
FACT
VALUE
VALUE
Input: Ordered a NEW product -
Received a USED product.
Having to leave the country this week, and my other BOSE headset gone out, I must use THIS product, which was used.
Extremely disappointed.
Output: TESTIMONY
TESTIMONY
VALUE
VALUE
Input: These lasted no time at all,
one speaker went out in less than one month.
Then the other speaker followed soon after.
Don't waste your money
and buy better quality headphones that will last.
Output: TESTIMONY
TESTIMONY
TESTIMONY
POLICY
POLICY

Task: You are given the following question, stance (Yes vs. No) towards this question and a type of reasoning (logos vs. pathos). Your task is to form a persuasive argument toward the question that supports the given stance based on the following type of reasoning.
Input: Question: Should Germany introduce the death penalty?
Stance: No
Reasoning: Emotional reasoning (pathos-oriented)
Output: Actually the death penalty is a legal means that as such is not practicable in Germany . Plus, many people think that a murderer has already decided on the life or death of another person . On the other hand, it is a much graver punishment to be imprisoned forever and be tortured by one's own thoughts than to be killed quickly and easily by an injection . Besides, a death would not be of any more use to those affected and their relatives than if the felon receives a long sentence 
Input: Question: Should the statutory retirement age remain at 63 years in the future?
Stance: No
Reasoning: Logical reasoning (logos-oriented)
Output: Due to the low birth rate, the proportion of the elderly population and with it pension costs are on the rise especially now, as there is a lack of skilled labour, it's irresponsible for the economy and prosperity to retire at 63. Even though the number of immigrants is constantly rising in Germany , without sufficient, well-qualified junior employees there is hardly a possibility for adequate pension financing .
Input: Question: Should public health insurance cover treatments in complementary and alternative medicine?
Stance: Yes
Reasoning: Emotional reasoning (pathos-oriented)
Output: Of course  health insurance companies should naturally cover alternative medical treatments ! Maybe not all practices and approaches that are lumped together under this term may have been proven in clinical trials but many people see proven relief of their symptoms and complaints by complementary medicine and such therapies can frequently result in an improvement in the condition through the active participation of the patient .
Input: Question: Should intelligence services be regulated more tightly by parliament?
Stance: Yes
Reasoning: Logical reasoning (logos-oriented)
Output: Intelligence services must be monitored more closely to prevent abuse . While a certain level of secrecy is the basis for the work of intelligence services , intelligence services intervene massively in the privacy of citizens . In fact, their tools, data and expertise have been used to keep us under surveillance for a long time .

Task: Given an argument consisting of a claim and a reason, select the correct warrant that explains reasoning of this particular argument. There are only two options given and only one answer is correct.
Input: Topic: Does Turkey Still Belong in NATO?
Additional Info: Given President Erdogan's record on human rights and how his focus on the Kurdish minority has interfered with his fight against ISIS, is he a reliable ally?
Claim: Turkey doesn't belong to NATO
Reason: Erdogan has a history or repressing civil rights and violence.
Warrant 1: NATO is an organization that routinely exposes violations of civil rights
Warrant 2: NATO is an organization that routinely violates civil rights
Output: NATO is an organization that routinely exposes violations of civil rights
Input: Topic: Is Extreme Parenting Effective?
Additional Info: Does strict control of a child's life lead to greater success or can it be counterproductive?
Claim: Extreme parenting is counterproductive
Reason: Extreme parenting doesn't necessarily raise a compassionate child.
Warrant 1: compassionate adults are often more successful than their peers
Warrant 2: compassionate adults are often less successful than their peers
Output: compassionate adults are often more successful than their peers
Input: Topic: Should Salt Have a Place at the Table?
Additional Info: More restaurants have been doing without salt shakers. Some chefs say that's sensible. Some diners find it annoying.
Claim: Salt should have a place at the table
Reason: Salt shakers permit diners to control the amount of salt in their food.
Warrant 1: no one goes out to eat if they want control over what they're shoving into their mouth
Warrant 2: people go out to eat when they want control over what they're shoving into their mouth
Output: people go out to eat when they want control over what they're shoving into their mouth

Task: Extract the Toulmin components (Premise, Claim, Backing, Refutation and Rebuttal) from the given argument. The output should be in the format: "Premise: <premise> --> Claim: <claim>" or "Refutation: <refutation> --> Rebuttal: <rebuttal>" or "Rebuttal: <rebuttal> --> Claim: <claim>" or "Backing: <backing>"
Input: There's no way in he** I would send my kid to a public school in today's times.  The public school that I went to now has gangs that walk the hallways.  Just about every week cops are called to this school because of either a shooting/stabbing.  The neighboring county public schools are the same.  Your public schools in parts of this country have become a cess pool of degenerates and with parents that are about the same.  I would work two jobs if needed to send my kid to a private school (one that I had done a lot of background checking into first).  Not saying that private schools don't have problems either but none to the ratio that public schools have.
Output: Premise: The public school that I went to now has gangs that walk the hallways.  Just about every week cops are called to this school because of either a shooting/stabbing.  The neighboring county public schools are the same.  Your public schools in parts of this country have become a cess pool of degenerates and with parents that are about the same. --> Claim: There's no way in he** I would send my kid to a public school in today's times.
Input: There is a time and a place for everything, including religion . Public School is not the place for religion, unless we are doing a Comparative Religions course, in which all views are examined. Prayer is something that needs to be and should be taught in the home, by parents. If you want children to pray at school , there are plenty of Catholic and other private schools available. Otherwise, any discussion of religion should be strictly technical.
Output: Rebuttal: we are doing a Comparative Religions course, in which all views are examined. --> Claim: Public School is not the place for religion,
Premise: Prayer is something that needs to be and should be taught in the home, by parents. --> Claim: Public School is not the place for religion,
Backing: There is a time and a place for everything, including religion .
Backing: If you want children to pray at school , there are plenty of Catholic and other private schools available.
Input: have three children with kid's of the their own, one moved to be able to send his kids to a private school, the other home school's her children, while my other children still go to a public school....but they're father is a teacher....thank God!!  public education is for the most part terrible,and it's too bad that our government had to step in and ruin what was working!!  maybe someday we'll elect official's that give the people what they really want, instead of slamming they're own agenda's down our throat!!!
Output: Premise: and it's too bad that our government had to step in and ruin what was working!! --> Claim: public education is for the most part terrible,
Backing: I have three children with kid's of the their own, one moved to be able to send his kids to a private school, the other home school's her children, while my other children still go to a public school....but they're father is a teacher....thank God!!

Task: Identify the reasons in the given argumentative text.
Input: What is more important is that abortion is the choice of the woman, because she has the child. It is best if at the beginning of pregnancy abortion, because that is the child is not large. Just before the birth abortion, I find it disgusting, because the child has to a large extent to be born perfect. So I think it's good as long as it is the beginning of pregnancy, but just before giving birth.
Output: Reason 0: What is more important is that abortion is the choice of the woman, because she has the child.
Reason 1: So I think it's good as long as it is the beginning of pregnancy, but just before giving birth.
Reason 2:  It is best if at the beginning of pregnancy abortion, because that is the child is not large
Input: Marijuana being illegal is part of the reason why people do it. It is a stupid reason, but a reason nonetheless. Legalizing marijuana, could potentially lower the number of users. The thrill of the chance of being caught gives them a high that's intensified by the drug. It outta just be legalized, and the government can tax the hell out of it like with cigarettes. It's your choice if you're going to do it or not.
Output: Reason 0: It's your choice if you're going to do it or not.
Reason 1: Marijuana being illegal is part of the reason why people do it.
Reason 2: It outta just be legalized, and the government can tax the hell out of it like with cigarettes.
Reason 3: Legalizing marijuana, could potentially lower the number of users.
Input: Hell no! His policies are crap and I can't wait for the courts to overturn his health care bill crap ;)
Output: Reason 0: I can't wait for the courts to overturn his health care bill crap
Reason 1: His policies are crap

Task: Produce a short text that argues for or against the following debate topic.
Input: Is hunting good or bad for the environment?
Output: Hunting is good for the environment because overpopulated species can be thinned out. Getting rid of an overpopulation enables the smaller animals in the food chain to grow. It also allows for nature to take back the woods and grasslands, which will also enable other wildlife to grow. Some people may object to hunting on the basis of humane treatment of animals. Animals do not necessarily feel pain as we do, and if there are not enough resources to sustain an overpopulation, then a quick bullet or arrow is better than slow starvation.
Input: Should car drivers be strictly prohibited from using cell phones?
Output: Using a cell phone while driving is extremely dangerous. Any time a driver is focused on a handheld device, they are not paying attention to the road. This can lead to accidents, which can cause serious injury and even death. Some people might argue that they could miss an important call while driving, but it is far better to pull to the side of the road and take the call rather than to keep driving. Prohibiting car drivers from using their cell phones while driving would go a long way to making our roads safer.
Input: Do violent video games cause people to act out violently?
Output: Violent video games do not cause people to act out violently. There are those that say that violence in video games exposes children to hurting people which could cause them to act those things out in reality. I disagree. In fact, I believe the evidence shows that the opposite is the case. Children, especially young boys who have an outlet for the expression that involves simulated events and not real ones helps them to learn about and work through important themes like violence. If there was not a medium for children to act out these situations in harmless ways they might experiment in reality. This would have even more disastrous consequences than a few hours gaming.

Task:'''


#Task: Given a controversial topic and a representation of the audience’s beliefs, generate a claim that is both relevant to the topic and matches the beliefs.
#Input: 
#Topic: abortion is wrong
#Beliefs:
#Abortion: Pro
#Affirmative Action: Con
#Animal Rights: Pro
#Barack Obama: No Opinion
#Border Fence: No Opinion
#Capitalism: Pro
#Civil Unions: Undecided
#Death Penalty: Pro
#Drug Legalization: Pro
#Electoral College: Undecided
#Environmental Protection: Pro
#Estate Tax: Undecided
#European Union: Undecided
#Euthanasia: Pro
#Federal Reserve: Con
#Flat Tax: Undecided
#Free Trade: Undecided
#Gay Marriage: Pro
#Global Warming Exists: Con
#Globalization: Pro
#Gold Standard: Undecided
#Gun Rights: Undecided
#Homeschooling: Undecided
#Internet Censorship: Con
#Iran-Iraq War: Undecided
#Labor Union: Undecided
#Legalized Prostitution: Pro
#Medicaid & Medicare: Undecided
#Medical Marijuana: Pro
#Military Intervention: Undecided
#Minimum Wage: Pro
#National Health Care: Undecided
#National Retail Sales Tax: Undecided
#Occupy Movement: Undecided
#Progressive Tax: Undecided
#Racial Profiling: Pro
#Redistribution: Undecided
#Smoking Ban: Undecided
#Social Programs: Undecided
#Social Security: Undecided
#Socialism: Con
#Stimulus Spending: Undecided
#Term Limits: Con
#Torture: Undecided
#United Nations: Undecided
#War in Afghanistan: Undecided
#War on Terror: Undecided
#Welfare: Undecided
#Output: The only people that possess the power to answer this question is women . First , the answer to whether or not abortion should be LEGAL is an undoubtable YES since this is a free country . Now , the answer to whether or not abortion is WRONG is arbitrary and relative , based upon your values and the source of such values ( I.E . Religion , agnosticism , divergent values ).
#Input: Topic: has president obama done a good job?
#Beliefs:
#Abortion: Pro
#Affirmative Action: No Opinion
#Animal Rights: Pro
#Barack Obama: Pro
#Border Fence: No Opinion
#Capitalism: Pro
#Civil Unions: No Opinion
#Death Penalty: Undecided
#Drug Legalization: Undecided
#Electoral College: Undecided
#Environmental Protection: Pro
#Estate Tax: No Opinion
#European Union: Pro
#Euthanasia: Pro
#Federal Reserve: Pro
#Flat Tax: Con
#Free Trade: Pro
#Gay Marriage: Pro
#Global Warming Exists: Pro
#Globalization: Pro
#Gold Standard: Con
#Gun Rights: Con
#Homeschooling: No Opinion
#Internet Censorship: Con
#Iran-Iraq War: Con
#Labor Union: Pro
#Legalized Prostitution: Pro
#Medicaid & Medicare: No Opinion
#Medical Marijuana: Pro
#Military Intervention: Pro
#Minimum Wage: Pro
#National Health Care: Pro
#National Retail Sales Tax: No Opinion
#Occupy Movement: Con
#Progressive Tax: Pro
#Racial Profiling: Undecided
#Redistribution: Pro
#Smoking Ban: Con
#Social Programs: Pro
#Social Security: Pro
#Socialism: Con
#Stimulus Spending: Pro
#Term Limits: Pro
#Torture: Con
#United Nations: Pro
#War in Afghanistan: Pro
#War on Terror: Pro
#Welfare: Pro
#Output: Tact , leadership , and insight Prevented an economic depression , passed Universal Healthcare , killed Osama Bin Laden , got General Motors back on track , all while facing insidious opposition from Republicans who refused to meet him halfway . It 's absurd to say that he did n't do a good job.
#Input: Topic: should there be more animal protection laws?
#Beliefs:
#Abortion: Con
#Affirmative Action: Con
#Animal Rights: Con
#Barack Obama: No Opinion
#Border Fence: No Opinion
#Capitalism: Undecided
#Civil Unions: Undecided
#Death Penalty: Undecided
#Drug Legalization: Undecided
#Electoral College: Undecided
#Environmental Protection: Pro
#Estate Tax: Undecided
#European Union: Undecided
#Euthanasia: Undecided
#Federal Reserve: Undecided
#Flat Tax: Undecided
#Free Trade: Undecided
#Gay Marriage: Con
#Global Warming Exists: Pro
#Globalization: Undecided
#Gold Standard: Undecided
#Gun Rights: Pro
#Homeschooling: Pro
#Internet Censorship: Con
#Iran-Iraq War: Con
#Labor Union: Undecided
#Legalized Prostitution: Pro
#Medicaid & Medicare: Pro
#Medical Marijuana: Undecided
#Military Intervention: Undecided
#Minimum Wage: Pro
#National Health Care: Undecided
#National Retail Sales Tax: Undecided
#Occupy Movement: Con
#Progressive Tax: Undecided
#Racial Profiling: Undecided
#Redistribution: Undecided
#Smoking Ban: Pro
#Social Programs: Pro
#Social Security: Pro
#Socialism: Undecided
#Stimulus Spending: Undecided
#Term Limits: Undecided
#Torture: Undecided
#United Nations: Undecided
#War in Afghanistan: Undecided
#War on Terror: Undecided
#Welfare: Undecided
#Output: Primarily for the normal person : There already exist many laws and regulations that exist for the possession of animals and the way in which they should be treated . I met a lady who claimed that she had being charged for possessing a 'pet ' squirrel that she adopted when its mother could not be found . This story is not quite verifiable , but the fact that there are many instances in which animals receive better living conditions and apparently more protection that human beings.

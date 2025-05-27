import re
from abc import abstractmethod


class EntryFormat:
    def __init__(self, ruleList: list):
        self.ruleList = ruleList

    def __call__(self, entry: str):
        for rule in self.ruleList:
            match, output = rule(entry)
            if match:
                return {} if output == None else output
        return {}


class EntryRule:
    @abstractmethod
    def __call__(self, entry: str) -> (bool, dict):
        pass


class AlwaysMatchedRule(EntryRule):
    def __call__(self, entry: str) -> (bool, dict):
        return (True, {})


class SkipRule(AlwaysMatchedRule):
    def __call__(self, entry: str) -> (bool, dict):
        return (
            True,
            {
                "entry": entry,
            },
        )


class AlwaysDismatchedRule(EntryRule):
    def __call__(self, entry: str) -> (bool, dict):
        return (False, {})


class PrefixRule(EntryRule):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, entry: str) -> (bool, dict):
        return (entry.strip().startswith(self.prefix), {})


PrefixIgnoreRule = PrefixRule  # alias


class PrefixSkipRule(PrefixRule):
    def __call__(self, entry: str) -> (bool, dict):
        return (entry.strip().startswith(self.prefix), {"entry": entry})


class RegExRule(EntryRule):
    def __init__(self, regex: str, index_group_pairs: dict):
        self.regex = re.compile(regex)
        self.index_group_pairs = index_group_pairs

    def __call__(self, entry: str) -> (bool, dict):
        match = self.regex.search(entry)
        if match == None:
            return (False, {})
        output = {}
        for index, group in self.index_group_pairs.items():
            if group == -1:
                output[index] = entry
            else:
                output[index] = match.group(group)
        return (True, output)


class RuleGroup:
    def __init__(
        self,
        entrance_rule: EntryRule,
        exit_rule: EntryRule,
        ruleList: list,
        dismatch_policy="exit_rule",
    ):
        self.entrance_rule = entrance_rule
        self.exit_rule = exit_rule
        self.ruleList = ruleList
        self.dismatch_policy = dismatch_policy

    def access(self, entry: str) -> bool:
        return self._enter(entry)

    def _enter(self, entry: str) -> bool:
        return self.entrance_rule(entry)[0]

    def _exit(self, entry: str) -> bool:
        return self.exit_rule(entry)[0]

    def _dismatch(self, entry: str) -> bool:
        def dismatch_policy_exit_rule(entry: str):
            return self._exit(entry)

        def dismatch_policy_exit(entry: str):
            return True

        def dismatch_policy_stay(entry: str):
            return False

        dismatch_policy_set = {
            "default": dismatch_policy_exit_rule,
            "exit_rule": dismatch_policy_exit_rule,
            "exit": dismatch_policy_exit,
            "stay": dismatch_policy_stay,
        }

        dismatch_policy = dismatch_policy_set.get(
            self.dismatch_policy, dismatch_policy_set["default"]
        )

        return dismatch_policy(entry)

    def __call__(self, entry: str):
        for rule in self.ruleList:
            match, output = rule(entry)
            if match:
                return ({} if output == None else output, self._exit(entry))

        return ({}, self._dismatch(entry))


class EntryFormatWithRuleGroups(EntryFormat):
    def __init__(self, ruleGroups: list):
        self.ruleGroups = ruleGroups
        self._nowGroup = None

    def inGroup(self) -> bool:
        return self._nowGroup != None

    def quitGroup(self):
        self._nowGroup = None

    def __call__(self, entry: str):
        if self._nowGroup == None:
            for ruleGroup in self.ruleGroups:
                if ruleGroup.access(entry):
                    self._nowGroup = ruleGroup
                    break
            else:
                # no available rule group
                return {}

        output, exited = self._nowGroup(entry)
        if exited:
            self.quitGroup()
        return output
